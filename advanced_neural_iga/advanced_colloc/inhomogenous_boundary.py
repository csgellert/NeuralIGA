import collocation_WEB as cWEB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from contextlib import contextmanager
from pathlib import Path
import Geomertry
import PDE_testcases
import torch
import network_defs as netdefs
from typing import Any, Dict, Optional
torch.set_default_dtype(torch.float64)

model = netdefs.load_test_model("SIREN_pentagon_MMSDF", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})


@contextmanager
def _pinned_function_case(function_case):
    """Pin PDE_testcases.FUNCTION_CASE to ``function_case`` for the duration of
    the with-block, restoring it afterward.

    Every PDE_testcases value function (dirichletBoundary_vectorized,
    load_function_vectorized, dirichletBoundary_side_vectorized, ...) and the
    case-11 disc clip in ``_DistanceCombinerBase.forward`` read the
    module-global FUNCTION_CASE rather than taking it as an argument.
    ``collocation_WEB.run_example`` only pins it for the duration of the
    solve, then restores whatever it was *before* that call -- so a
    reconstruction/plotting call made afterward (e.g. from a notebook cell)
    is at the mercy of whatever case an earlier cell happened to leave
    active. Without this, case 12 could silently get case 11's unit-disc
    clip applied to its weight function (or vice versa) during
    reconstruction, even though the solve itself was correct.
    """
    old_case = PDE_testcases.FUNCTION_CASE
    PDE_testcases.FUNCTION_CASE = function_case
    try:
        yield
    finally:
        PDE_testcases.FUNCTION_CASE = old_case


# =============================================================================
# SIDE-DISTANCE COMBINATORS
# =============================================================================
# Each combinator turns a per-side signed-distance model (points -> (N, num_sides))
# into a single scalar WEB-spline weight function w(x,y): positive inside the
# domain, negative outside, zero on the boundary. FUNCTION_CASE == 11 additionally
# clips w to the unit disc, since the 5-sided model only approximates the circular
# boundary between its sampled corners and can bulge slightly outside it.

class _DistanceCombinerBase(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _combine(self, distances: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x):
        w = self._combine(self.model(x))
        if PDE_testcases.FUNCTION_CASE == 11:
            outside_disc = x[:, 0] ** 2 + x[:, 1] ** 2 > 1.0
            w = torch.where(outside_disc, torch.tensor(-1.0, dtype=x.dtype, device=x.device), w)
        return w


class sharp_min_distance_function(_DistanceCombinerBase):
    """Exact minimum across side distances."""
    def _combine(self, distances):
        return torch.min(distances, dim=1).values


class sharp_min_distance_function_hollig(_DistanceCombinerBase):
    """Exact minimum, post-processed with the Höllig weight transform."""
    def __init__(self, model, delta=0.1, gamma=3.0):
        super().__init__(model)
        self.delta = delta
        self.gamma = gamma

    def _combine(self, distances):
        d = torch.min(distances, dim=1).values
        return Geomertry.hollig_transform(d=d, delta=self.delta, gamma=self.gamma)


class smooth_min_distance_function(_DistanceCombinerBase):
    """Smooth (log-sum-exp) minimum across side distances."""
    def __init__(self, model, alpha=10.0):
        super().__init__(model)
        self.alpha = alpha

    def _combine(self, distances):
        return -torch.logsumexp(-self.alpha * distances, dim=1) / self.alpha


class R_func_min(_DistanceCombinerBase):
    """Rvachev R-function (smooth, zero-preserving) union of side distances."""
    def _combine(self, distances):
        union = distances[:, 0]
        for i in range(1, distances.shape[1]):
            d2 = distances[:, i]
            union = union + d2 - torch.sqrt(union ** 2 + d2 ** 2)
        return union


def _eval_side_distances(model, points, num_sides, track_grad=False):
    """Evaluate the per-side distance model, with an autograd graph only when needed."""
    if track_grad:
        distances = model(points)
    else:
        with torch.no_grad():
            distances = model(points)
    assert distances.shape[1] == num_sides, f"Expected {num_sides} side distances, got {distances.shape[1]}."
    return distances


def _ratio_side_param(distances):
    """FUNCTION_CASE 11's original ('distance_ratio') side parameter: for side
    i, the ratio of that side's own distance to (that distance + the distance
    of the side two positions ahead), cyclically. This is a heuristic, not an
    actual nearest-point projection -- kept exactly as originally implemented
    since case 11's validated boundary values depend on it (see
    PDE_testcases.CASE_POLYGON_GEOMETRY's side_placement docs).
    """
    num_sides = distances.shape[1]
    s_coords = torch.zeros_like(distances)
    distances_ext = torch.cat([distances, distances[:, :2]], dim=1)
    for i in range(1, num_sides + 1):
        s_coords[:, i - 1] = distances_ext[:, i - 1] / (distances_ext[:, i - 1] + distances_ext[:, i + 1])
    return torch.cat([s_coords[:, -1:], s_coords[:, :-1]], dim=1)


# =============================================================================
# PENTAGON BOUNDARY GEOMETRY AND PER-SIDE DIRICHLET DATA
# =============================================================================
# Geometry (radius/center/rotation/bulge/curved sides) and the prescribed
# per-side Dirichlet data both live in PDE_testcases.py (CASE_POLYGON_GEOMETRY
# and CASE_SIDE_ZERO_DIRICHLET / dirichletBoundary_side_vectorized), so adding a
# case or reassigning boundary values is a one-place edit there.

_LIFTING_CACHE: Dict[Any, torch.Tensor] = {}


def _build_polygon_boundary_data(function_case: int, num_sides: int) -> Dict[str, Any]:
    """Build per-side control-point curves for a pentagon FUNCTION_CASE.

    Straight sides are stored as 2-point line segments; sides listed in the
    case's ``curved_side_indices`` (PDE_testcases.CASE_POLYGON_GEOMETRY) are
    stored as 4-point cubic Bezier-equivalent spans bulging inward.
    """
    if num_sides != 5:
        raise ValueError("Pentagon boundary geometry requires num_sides=5.")

    params = PDE_testcases.get_case_polygon_geometry(function_case)
    radius = float(params["radius"])
    center = params["center"]
    rotation = float(params["rotation"])
    bulge = float(params["bulge"])
    samples_per_side = int(params["samples_per_side"])
    curved_set = set(int(i) for i in params["curved_side_indices"])

    angles = rotation + (2.0 * np.pi / num_sides) * np.arange(num_sides, dtype=np.float64)
    vertices = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])

    side_curves = []
    for side_idx in range(num_sides):
        v0, v1 = vertices[side_idx], vertices[(side_idx + 1) % num_sides]
        edge = v1 - v0
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 1e-14:
            raise ValueError(f"Degenerate pentagon edge for FUNCTION_CASE={function_case}.")

        if side_idx in curved_set:
            n_in = np.array([-edge[1], edge[0]]) / edge_len  # inward normal (CCW polygon)
            curv = bulge * edge_len
            p1 = (2.0 / 3.0) * v0 + (1.0 / 3.0) * v1 + curv * n_in
            p2 = (1.0 / 3.0) * v0 + (2.0 / 3.0) * v1 + curv * n_in
            side_curves.append(np.vstack([v0, p1, p2, v1]))
        else:
            side_curves.append(np.vstack([v0, v1]))

    return {
        "side_curves": side_curves,
        "samples_per_side": samples_per_side,
        "cache_key": (function_case, radius, tuple(center), rotation, bulge,
                      tuple(sorted(curved_set)), samples_per_side),
    }


def _evaluate_side_curve(t, control_points):
    """Evaluate a straight (2 control points) or cubic Bezier-equivalent
    (4 control points) side span at parameters ``t`` in [0, 1]."""
    if control_points.shape[0] == 2:
        p0, p1 = control_points
        return p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    p0, p1, p2, p3 = control_points
    u = 1.0 - t
    b0, b1, b2, b3 = u ** 3, 3 * u ** 2 * t, 3 * u * t ** 2, t ** 3
    return (
        b0[:, None] * p0[None, :] + b1[:, None] * p1[None, :]
        + b2[:, None] * p2[None, :] + b3[:, None] * p3[None, :]
    )


def _sample_side_curves(boundary_data, dtype, device):
    """Sample each side curve into a polyline, cached per geometry/dtype/device."""
    cache_key = (boundary_data["cache_key"], str(device), str(dtype))
    cached = _LIFTING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    t_vals = torch.linspace(0.0, 1.0, boundary_data["samples_per_side"], dtype=dtype, device=device)
    side_samples = torch.stack(
        [
            _evaluate_side_curve(t_vals, torch.as_tensor(cp, dtype=dtype, device=device))
            for cp in boundary_data["side_curves"]
        ],
        dim=0,
    )
    _LIFTING_CACHE[cache_key] = side_samples
    return side_samples


def _project_points_to_sides(points, side_samples, chunk_size=32768):
    """Nearest-point projection of ``points`` onto each side's sampled polyline.

    Returns
    -------
    projected : Tensor (N, num_sides, 2)
        Closest sampled point on each side.
    s_param : Tensor (N, num_sides)
        Fractional index of the closest sample along that side, in [0, 1].
    """
    n_points = points.shape[0]
    n_sides, n_samples, _ = side_samples.shape
    dtype, device = points.dtype, points.device

    projected = torch.zeros(n_points, n_sides, 2, dtype=dtype, device=device)
    s_param = torch.zeros(n_points, n_sides, dtype=dtype, device=device)

    for side_idx in range(n_sides):
        curve = side_samples[side_idx]
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            diff = points[start:end, None, :] - curve[None, :, :]
            idx = torch.argmin(torch.sum(diff * diff, dim=2), dim=1)
            projected[start:end, side_idx, :] = curve[idx]
            if n_samples > 1:
                s_param[start:end, side_idx] = idx.to(dtype) / (n_samples - 1)

    return projected, s_param


def _side_points_and_values(function_case, num_sides, points, distances=None):
    """Locate, for each point and each side, the boundary point the Dirichlet
    value is evaluated at, and fetch that prescribed value.

    Which point on a side represents a query point is controlled by
    ``side_placement`` in PDE_testcases.CASE_POLYGON_GEOMETRY:
    - "distance_ratio": case 11's original scheme, placing the point via
      ``_ratio_side_param`` on the model's own raw ``distances`` (required
      in this mode).
    - "nearest_point": true nearest-point projection onto a sampled polyline
      of the side curve (works for curved sides too).

    ``PDE_testcases.dirichletBoundary_side_vectorized`` (like every value
    function in PDE_testcases.py) reads the module-global ``FUNCTION_CASE``
    rather than taking it as an argument, so it is pinned to the requested
    ``function_case`` here for the duration of the call. Without this, a
    caller that passes an explicit ``function_case`` while a *different* case
    is left active globally (e.g. from an earlier notebook cell or a previous
    ``run_example`` call, which restores the global to whatever it was
    *before* it ran) would silently get boundary values and zeroed sides for
    the wrong case.

    Returns
    -------
    side_points : Tensor (N, num_sides, 2)
    bnd_values : Tensor (N, num_sides)
    s_param : Tensor (N, num_sides)  position along each side, in [0, 1]
    """
    params = PDE_testcases.get_case_polygon_geometry(function_case)
    boundary_data = _build_polygon_boundary_data(function_case, num_sides)

    if params.get("side_placement", "nearest_point") == "distance_ratio":
        if distances is None:
            raise ValueError("side_placement='distance_ratio' requires the model's side distances.")
        s_param = _ratio_side_param(distances)
        side_points = torch.stack(
            [
                _evaluate_side_curve(s_param[:, i], torch.as_tensor(cp, dtype=points.dtype, device=points.device))
                for i, cp in enumerate(boundary_data["side_curves"])
            ],
            dim=1,
        )
    else:
        side_samples = _sample_side_curves(boundary_data, dtype=points.dtype, device=points.device)
        side_points, s_param = _project_points_to_sides(points, side_samples)

    # Kept as torch ops throughout (no numpy round-trip): for "distance_ratio"
    # placement, side_points depends smoothly on the model's own distances, and
    # get_lifting_laplacian's autograd-based Laplacian needs that dependency
    # to stay differentiable. ("nearest_point" placement's argmin projection
    # has ~zero gradient w.r.t. points anyway, so this is a no-op for case 12.)
    n_points = points.shape[0]
    pts_flat = side_points.reshape(-1, 2)
    side_idx_flat = torch.arange(num_sides, device=points.device).repeat(n_points)
    with _pinned_function_case(function_case):
        bnd_values = PDE_testcases.dirichletBoundary_side_vectorized(
            pts_flat[:, 0], pts_flat[:, 1], side_idx_flat,
        )
    bnd_values = bnd_values.reshape(n_points, num_sides)
    return side_points, bnd_values, s_param


# =============================================================================
# LIFTING (NON-HOMOGENEOUS DIRICHLET) BLENDING
# =============================================================================

def _blend_lifting_values(d, bnd_values, s, smoothness=2, blend_method="inverse_distance", eps=1e-12):
    """Blend per-side boundary values into one lifting value per point, weighted
    by the per-side distances ``d``. Points with any non-positive side distance
    (outside the domain) get lifting value 0.

    blend_method:
    - 'inverse_distance': inverse-distance-weighted average of the sides.
    - 'wachspress': corner (vertex) values blended by adjacent-side weights.
    - 'coons': corner values interpolated along each edge by ``s``, then
      blended by inverse-distance side weights.
    """
    valid_mask = torch.all(d > 0, dim=1)
    ud = torch.zeros(d.shape[0], dtype=d.dtype, device=d.device)
    if not torch.any(valid_mask):
        return ud, valid_mask

    ds, bs, ss = d[valid_mask], bnd_values[valid_mask], s[valid_mask]

    if blend_method == "inverse_distance":
        w = 1.0 / (ds ** smoothness + eps)
        ud_vals = torch.sum(w * bs, dim=1) / (torch.sum(w, dim=1) + eps)

    elif blend_method in ("wachspress", "coons"):
        # Corner i sits between side (i-1) and side i; roll(shift=1) gathers side (i-1).
        left_bnd = torch.roll(bs, shifts=1, dims=1)
        left_dist = torch.roll(ds, shifts=1, dims=1)
        corner_values = (left_bnd * ds + bs * left_dist) / (left_dist + ds + eps)

        if blend_method == "wachspress":
            vertex_weights = 1.0 / ((left_dist + eps) * (ds + eps))
            vertex_weights = vertex_weights / (torch.sum(vertex_weights, dim=1, keepdim=True) + eps)
            ud_vals = torch.sum(vertex_weights * corner_values, dim=1)
        else:
            # Edge i interpolates between its left corner (i) and right corner (i+1).
            right_corner_values = torch.roll(corner_values, shifts=-1, dims=1)
            edge_values = (1.0 - ss) * corner_values + ss * right_corner_values
            side_weights = 1.0 / (ds ** smoothness + eps)
            side_weights = side_weights / (torch.sum(side_weights, dim=1, keepdim=True) + eps)
            ud_vals = torch.sum(side_weights * edge_values, dim=1)
    else:
        raise ValueError(f"Unknown blend_method '{blend_method}'.")

    ud[valid_mask] = ud_vals
    return ud, valid_mask


def _laplacian_from_autograd(values, points):
    grad_vals = torch.autograd.grad(
        outputs=values.sum(),
        inputs=points,
        create_graph=True,
        retain_graph=True,
    )[0]

    lap_vals = torch.zeros_like(values)
    for axis_idx in range(points.shape[1]):
        second_axis = torch.autograd.grad(
            outputs=grad_vals[:, axis_idx].sum(),
            inputs=points,
            retain_graph=axis_idx < points.shape[1] - 1,
        )[0][:, axis_idx]
        lap_vals = lap_vals + second_axis
    return lap_vals


def _compute_lifting_core(
    model,
    function_case,
    num_sides,
    points,
    smoothness=2,
    blend_method="inverse_distance",
    eps=1e-12,
    return_laplacian=False,
    track_grad=None,
):
    """``track_grad`` defaults to ``return_laplacian`` (the value needs no
    autograd graph unless the Laplacian is requested), but callers that need
    a differentiable ``ud`` without the Laplacian (e.g. get_lifting_gradient)
    can force it on explicitly.
    """
    if track_grad is None:
        track_grad = return_laplacian
    if track_grad and not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)

    d = _eval_side_distances(model, points, num_sides, track_grad=track_grad)
    _, bnd_values, s_param = _side_points_and_values(function_case, num_sides, points, distances=d)

    ud, _ = _blend_lifting_values(
        d=d, bnd_values=bnd_values, s=s_param,
        smoothness=smoothness, blend_method=blend_method, eps=eps,
    )

    if return_laplacian:
        return ud, _laplacian_from_autograd(values=ud, points=points)
    return ud


def get_lifting(model, function_case, num_sides, points, smoothness=2):
    """Inverse-distance-weighted blend of the prescribed side Dirichlet values."""
    return _compute_lifting_core(model, function_case, num_sides, points, smoothness,
                                  blend_method="inverse_distance")


def get_lifting_laplacian(model, function_case, num_sides, points, smoothness=2):
    """Like ``get_lifting``, plus the exact Laplacian of the lifting function
    (cheaper than materializing the full Hessian via autograd)."""
    return _compute_lifting_core(model, function_case, num_sides, points, smoothness,
                                  blend_method="inverse_distance", return_laplacian=True)


def get_lifting_polygon(model, function_case, num_sides, points, smoothness=2,
                         blend_method="wachspress", eps=1e-12, return_laplacian=False):
    """Polygon-aware lifting. blend_method: 'inverse_distance', 'wachspress', or 'coons'.

    If ``return_laplacian`` is True, returns ``(ud, lap_ud)`` with the exact
    Laplacian computed without materializing the full Hessian.
    """
    return _compute_lifting_core(model, function_case, num_sides, points, smoothness,
                                  blend_method=blend_method, eps=eps, return_laplacian=return_laplacian)


def get_lifting_gradient(model, function_case, num_sides, points, smoothness=2,
                          blend_method="inverse_distance", eps=1e-12):
    """Lifting value together with its gradient (d/dx, d/dy), via autograd.

    Returns
    -------
    ud : Tensor, shape (N,)
    dud : Tensor, shape (N, 2)
    """
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)
    ud = _compute_lifting_core(model, function_case, num_sides, points, smoothness,
                                blend_method=blend_method, eps=eps, return_laplacian=False, track_grad=True)
    dud = torch.autograd.grad(outputs=ud.sum(), inputs=points, create_graph=False, retain_graph=False)[0]
    return ud, dud


# =============================================================================
# PLOTTING / DIAGNOSTICS
# =============================================================================

def plot_case_11_poisson_samples(
    recon_info,
    model,
    csv_path=None,
    title_prefix="Function case 11",
    show: bool = True,
    filename: str = "poisson_samples_hom_1_e_3.csv",
    model_ms=None,
    lifting_method: str = "get_lifting",
    function_case: int = 11,
):
    """Plot ground truth, reconstructed solution, and absolute error for case 11.

    The CSV file must contain rows of the form x,y,u with a header line.
    The reconstructed solution is evaluated at the sample points using the
    collocation coefficients stored in recon_info.
    """
    if recon_info is None:
        raise ValueError("recon_info is required. Call collocation_WEB.run_example(..., return_coefficients=True).")

    csv_file = Path(csv_path) if csv_path is not None else Path(__file__).with_name(filename)
    data = np.genfromtxt(csv_file, delimiter=",", names=True)

    if data.size == 0:
        raise ValueError(f"No samples found in {csv_file}")

    pts_x = np.asarray(data["x"], dtype=np.float64).ravel()
    pts_y = np.asarray(data["y"], dtype=np.float64).ravel()
    gt = np.asarray(data["u"], dtype=np.float64).ravel()

    # model's forward (e.g. sharp_min_distance_function) reads the global
    # FUNCTION_CASE for its case-11 disc clip, so it must be pinned here --
    # see _pinned_function_case.
    with _pinned_function_case(function_case):
        wfct_phys = cWEB.NeuralWeightFunction(model=model, domain=None)
        pred = cWEB.reconstruct_collocation_at_points(
            pts_x, pts_y, recon_info, wfct_phys, model_ms=model_ms,
            lifting_method=lifting_method, function_case=function_case,
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    error = _plot_gt_pred_error_row(axes, pts_x, pts_y, gt, pred, title_prefix)

    if show:
        plt.show()

    return {
        "x": pts_x,
        "y": pts_y,
        "gt": gt,
        "pred": pred,
        "error": error,
        "csv_path": str(csv_file),
    }


def _masked_triangulation(pts_x, pts_y, edge_factor=3.0):
    """Delaunay triangulation of scattered points with long 'bridge' triangles masked out.

    Plain Delaunay triangulation always fills the *convex hull* of its
    points. For a non-convex domain -- e.g. case 12's mixed pentagon, which
    bulges *inward* on 3 sides -- that silently paints over the concave
    notches with extra triangles bridging across them, making tricontourf
    render what looks like a plain convex (straight-sided) pentagon instead
    of the true curved-side shape. Masking out triangles whose longest edge
    is much longer than typical (a standard alpha-shape-style heuristic)
    removes those bridging triangles without needing to know the domain's
    exact boundary.
    """
    tri = Triangulation(pts_x, pts_y)
    tx = pts_x[tri.triangles]
    ty = pts_y[tri.triangles]
    edge01 = np.hypot(tx[:, 0] - tx[:, 1], ty[:, 0] - ty[:, 1])
    edge12 = np.hypot(tx[:, 1] - tx[:, 2], ty[:, 1] - ty[:, 2])
    edge20 = np.hypot(tx[:, 2] - tx[:, 0], ty[:, 2] - ty[:, 0])
    max_edge = np.maximum(np.maximum(edge01, edge12), edge20)
    tri.set_mask(max_edge > edge_factor * np.median(max_edge))
    return tri


def _plot_gt_pred_error_row(axes_row, pts_x, pts_y, gt, pred, label, cmap_val="viridis", cmap_err="hot"):
    """Render one row of 3 panels (ground truth, reconstructed, absolute error)."""
    error = pred - gt
    triangulation = _masked_triangulation(pts_x, pts_y)
    panels = [
        (gt, f"{label} ground truth", cmap_val),
        (pred, f"{label} reconstructed", cmap_val),
        (np.abs(error), f"{label} absolute error", cmap_err),
    ]
    for ax, (values, title, cmap) in zip(axes_row, panels):
        contour = ax.tricontourf(triangulation, values, levels=50, cmap=cmap)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    return error


def _lifting_method_to_blend(lifting_method):
    """Map a reconstruct_collocation_at_points-style ``lifting_method`` name to
    the ``blend_method`` understood by get_lifting_gradient/get_lifting_polygon."""
    if lifting_method == "get_lifting":
        return "inverse_distance"
    if lifting_method in ("wachspress", "coons"):
        return lifting_method
    raise ValueError(
        f"Gradient/Laplacian reconstruction does not support lifting_method='{lifting_method}'. "
        "Use 'get_lifting', 'wachspress', 'coons', or 'none'."
    )


def _reconstruct_gradient_and_laplacian(
    pts_x, pts_y, recon_info, wfct_phys, model_ms, lifting_method, function_case, num_sides=5,
):
    """Reconstruct the gradient magnitude and Laplacian of the full collocation
    solution u = max(w, 0) * v + lifting at arbitrary physical points.

    The homogeneous WEB-spline part w*v is differentiated analytically via
    ``collocation_WEB.reconstruct_collocation_hessian_diag``. The lifting
    function's own contribution is also differentiated analytically, not
    numerically:
    - ``lifting_method="lightning"``: the lightning solution is (the real
      part of) an explicit holomorphic function built from elementary
      polynomial/Arnoldi and rational-pole terms, each of which has a known
      closed-form derivative -- see
      ``LightningLaplaceSolution._evaluate_complex_and_derivative`` and
      ``.evaluate_gradient``. Its Laplacian is exactly zero: a harmonic
      function's real part solves Delta(u)=0 by construction, independent of
      the boundary least-squares fit residual, so there is nothing to add
      for the Laplacian in this branch.
    - otherwise: autograd (get_lifting_gradient / get_lifting_polygon),
      using the same blend method the value reconstruction used, so the two
      stay consistent.
    """
    pts_x = np.asarray(pts_x, dtype=cWEB.NP_DTYPE).ravel()
    pts_y = np.asarray(pts_y, dtype=cWEB.NP_DTYPE).ravel()

    _, ux, uy, uxx, uyy = cWEB.reconstruct_collocation_hessian_diag(pts_x, pts_y, recon_info, wfct_phys)
    laplacian = uxx + uyy

    if lifting_method == "lightning":
        import lightning_laplace
        if function_case not in (11, 12):
            raise ValueError(
                "lifting_method='lightning' gradient/Laplacian reconstruction only "
                "supports function_case 11 or 12."
            )
        sol = lightning_laplace.solve_case_11(tolerance=1e-6) if function_case == 11 \
            else lightning_laplace.solve_case_12(tolerance=1e-6)
        _, lift_ux, lift_uy = sol.evaluate_gradient(pts_x, pts_y)
        ux = ux + lift_ux
        uy = uy + lift_uy
        # Delta(lifting) == 0 exactly (see docstring) -- laplacian unchanged.
    elif model_ms is not None and lifting_method != "none":
        blend_method = _lifting_method_to_blend(lifting_method)
        lift_points = torch.tensor(np.column_stack((pts_x, pts_y)), dtype=torch.float64)
        _, dud = get_lifting_gradient(model_ms, function_case, num_sides, lift_points.clone(), blend_method=blend_method)
        _, lap_ud = get_lifting_polygon(model_ms, function_case, num_sides, lift_points.clone(),
                                         blend_method=blend_method, return_laplacian=True)
        ux = ux + dud[:, 0].detach().cpu().numpy()
        uy = uy + dud[:, 1].detach().cpu().numpy()
        laplacian = laplacian + lap_ud.detach().cpu().numpy()

    grad_mag = np.sqrt(ux ** 2 + uy ** 2)
    return grad_mag, laplacian


def plot_case_12_poisson_samples(
    recon_info,
    model,
    csv_path=None,
    title_prefix="Function case 12",
    show: bool = True,
    filename: str = "poisson_samples_fun4_mixed_pentagon.csv",
    model_ms=None,
    lifting_method: str = "get_lifting",
    plot_gradient: bool = False,
    gradient_csv_path=None,
    gradient_filename: str = "poisson_samples_fun4_mixed_pentagon_gradient.csv",
    plot_laplacian: bool = False,
    function_case: int = 12,
    num_sides: int = 5,
):
    """Plot ground truth, reconstructed solution, and absolute error for case 12
    (mixed pentagon), like ``plot_case_11_poisson_samples``, plus two optional
    extra rows:

    - ``plot_gradient``: gradient-magnitude ground truth (read from
      ``gradient_csv_path``/``gradient_filename``, columns x,y,gradient_magnitude),
      reconstructed gradient magnitude, and their absolute error.
    - ``plot_laplacian``: reconstructed Laplacian of the solution against the
      analytic target ``-PDE_testcases.load_function_vectorized(x, y)`` for
      ``function_case`` (the PDE is -Delta(u) = f, so Delta(u) = -f), and their
      absolute error. There is no ground-truth file for this -- it is derived
      from the PDE's own right-hand side.

    When either extra row is plotted, its mean and max absolute error are also
    printed.

    The value CSV must contain columns x,y,u; the gradient CSV, x,y,gradient_magnitude.
    Both reconstructions use ``recon_info`` from
    ``collocation_WEB.run_example(..., return_coefficients=True)``.
    """
    if recon_info is None:
        raise ValueError("recon_info is required. Call collocation_WEB.run_example(..., return_coefficients=True).")

    csv_file = Path(csv_path) if csv_path is not None else Path(__file__).with_name(filename)
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No samples found in {csv_file}")

    pts_x = np.asarray(data["x"], dtype=np.float64).ravel()
    pts_y = np.asarray(data["y"], dtype=np.float64).ravel()
    gt = np.asarray(data["u"], dtype=np.float64).ravel()

    n_rows = 1 + int(plot_gradient) + int(plot_laplacian)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.8 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    result = {"x": pts_x, "y": pts_y, "gt": gt, "csv_path": str(csv_file)}

    # model's forward (e.g. sharp_min_distance_function) and PDE_testcases's
    # value functions all read the global FUNCTION_CASE, so everything that
    # touches the model or PDE_testcases is pinned to `function_case` here --
    # see _pinned_function_case.
    with _pinned_function_case(function_case):
        wfct_phys = cWEB.NeuralWeightFunction(model=model, domain=None)
        pred = cWEB.reconstruct_collocation_at_points(
            pts_x, pts_y, recon_info, wfct_phys, model_ms=model_ms,
            lifting_method=lifting_method, function_case=function_case,
        )
        error = _plot_gt_pred_error_row(axes[0], pts_x, pts_y, gt, pred, title_prefix)
        result["pred"] = pred
        result["error"] = error

        row = 1
        if plot_gradient:
            grad_csv_file = Path(gradient_csv_path) if gradient_csv_path is not None else Path(__file__).with_name(gradient_filename)
            grad_data = np.genfromtxt(grad_csv_file, delimiter=",", names=True)
            if grad_data.size == 0:
                raise ValueError(f"No samples found in {grad_csv_file}")

            grad_pts_x = np.asarray(grad_data["x"], dtype=np.float64).ravel()
            grad_pts_y = np.asarray(grad_data["y"], dtype=np.float64).ravel()
            grad_gt = np.asarray(grad_data["gradient_magnitude"], dtype=np.float64).ravel()

            grad_pred, _ = _reconstruct_gradient_and_laplacian(
                grad_pts_x, grad_pts_y, recon_info, wfct_phys, model_ms, lifting_method, function_case, num_sides,
            )
            grad_error = _plot_gt_pred_error_row(
                axes[row], grad_pts_x, grad_pts_y, grad_gt, grad_pred, f"{title_prefix} |grad u|",
            )
            print(f"Gradient magnitude: mean abs error = {np.mean(np.abs(grad_error)):.6e}, "
                  f"max abs error = {np.max(np.abs(grad_error)):.6e}")

            result.update({
                "grad_x": grad_pts_x, "grad_y": grad_pts_y,
                "grad_gt": grad_gt, "grad_pred": grad_pred, "grad_error": grad_error,
                "gradient_csv_path": str(grad_csv_file),
            })
            row += 1

        if plot_laplacian:
            _, lap_pred = _reconstruct_gradient_and_laplacian(
                pts_x, pts_y, recon_info, wfct_phys, model_ms, lifting_method, function_case, num_sides,
            )
            lap_target = -PDE_testcases.load_function_vectorized(pts_x, pts_y)

            lap_error = _plot_gt_pred_error_row(
                axes[row], pts_x, pts_y, lap_target, lap_pred, f"{title_prefix} Delta(u)",
            )
            print(f"Laplacian: mean abs error = {np.mean(np.abs(lap_error)):.6e}, "
                  f"max abs error = {np.max(np.abs(lap_error)):.6e}")

            result.update({
                "laplacian_target": lap_target, "laplacian_pred": lap_pred, "laplacian_error": lap_error,
            })

    if show:
        plt.show()

    return result


def plot_lifting_function(
    model=model,
    function_case: int = 11,
    num_sides: int = 5,
    N: int = 200,
    extent=None,
    cmap: str = 'viridis',
    show: bool = True,
):
    """Plot the lifting function u_d computed by get_lifting over the domain.

    Parameters:
    - model: per-side distance model used by get_lifting
    - function_case: integer case (11 for pentagon lifting)
    - num_sides: number of polygon sides used by the lifting logic
    - N: grid resolution per dimension
    - extent: (xmin, xmax, ymin, ymax). If None uses PDE_testcases.get_domain_for_case(function_case)
    - cmap: matplotlib colormap
    - show: whether to call plt.show()

    Returns: (X, Y, U_lift) numpy arrays
    """
    if extent is None:
        domain = PDE_testcases.get_domain_for_case(function_case)
        xmin, xmax = domain['x1'], domain['x2']
        ymin, ymax = domain['y1'], domain['y2']
    else:
        xmin, xmax, ymin, ymax = extent

    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    pts = np.column_stack([X.ravel(), Y.ravel()])
    pts_t = torch.tensor(pts, dtype=torch.float64)

    with torch.no_grad():
        ud_t = get_lifting(model=model, function_case=function_case, num_sides=num_sides, points=pts_t)

    ud = ud_t.detach().cpu().numpy().reshape(X.shape)

    plt.figure(figsize=(7, 6))
    im = plt.contourf(X, Y, ud, levels=60, cmap=cmap)
    plt.colorbar(im, label='Lifting u_d')
    plt.title(f'Lifting function (case {function_case})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')

    if show:
        plt.show()

    return X, Y, ud


if __name__ == "__main__":
    # rotation=0.0 must match CASE_POLYGON_GEOMETRY[11] in PDE_testcases.py: the
    # distance model's side order and the prescribed per-side Dirichlet data
    # must refer to the same physical pentagon.
    model_MS = Geomertry.AnaliticalDistancePentagon_SideDistances(rotation=0.0)
    model = sharp_min_distance_function(model_MS)
    #model = smooth_min_distance_function(model_MS, alpha=20.0)
    #model = R_func_min(model=model_MS)
    get_lifting(model_MS, function_case=11, num_sides=5, points=torch.tensor([[-0.80901699, 0.0], [0.0, -0.80901699], [0, 0]]))

    # Plot the distance function over [-1,1]x[-1,1]
    X = np.linspace(-1.1, 1.1, 100)
    Y = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(X, Y)
    points = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float64)
    distances = model(points).detach().numpy().reshape(X.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, distances, levels=50, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.contour(X, Y, distances, levels=[0], colors='red', linewidths=2)
    plt.title('Smooth Minimum Distance Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

    plot_lifting_function(model=model_MS, function_case=11, num_sides=5, N=200, extent=(-1, 1, -1, 1), cmap='viridis', show=True)
