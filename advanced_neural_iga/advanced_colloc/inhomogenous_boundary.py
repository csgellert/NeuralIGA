import collocation_WEB as cWEB
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Geomertry
import PDE_testcases
import torch
import network_defs as netdefs
from typing import Any, Dict, Optional
torch.set_default_dtype(torch.float64)

model = netdefs.load_test_model("SIREN_pentagon_MMSDF", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})


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
    old_case = PDE_testcases.FUNCTION_CASE
    PDE_testcases.FUNCTION_CASE = function_case
    try:
        bnd_values = PDE_testcases.dirichletBoundary_side_vectorized(
            pts_flat[:, 0], pts_flat[:, 1], side_idx_flat,
        )
    finally:
        PDE_testcases.FUNCTION_CASE = old_case
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
):
    if return_laplacian and not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)

    d = _eval_side_distances(model, points, num_sides, track_grad=return_laplacian)
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

    wfct_phys = cWEB.NeuralWeightFunction(model=model, domain=None)
    pred = cWEB.reconstruct_collocation_at_points(pts_x, pts_y, recon_info, wfct_phys, model_ms=model_ms, lifting_method=lifting_method)
    error = pred - gt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    plots = [
        (gt, f"{title_prefix} ground truth", "viridis"),
        (pred, f"{title_prefix} reconstructed solution", "viridis"),
        (np.abs(error), f"{title_prefix} absolute error", "hot"),
    ]

    for ax, (values, title, cmap) in zip(axes, plots):
        contour = ax.tricontourf(pts_x, pts_y, values, levels=50, cmap=cmap)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

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
