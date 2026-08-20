import collocation_WEB as cWEB
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Geomertry
import mesh
import PDE_testcases
import torch
import SDF
import network_defs as netdefs
torch.set_default_dtype(torch.float64)

#model = netdefs.Siren(architecture=[2, 256, 256, 256, 5], first_omega_0=80, hidden_omega_0=120.0, outermost_linear=True)
model = netdefs.load_test_model("SIREN_pentagon_MMSDF", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})

class sharp_min_distance_function(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        distances = self.model(x)
        # Exact sharp minimum across the side distances.
        sharp_min_distances = torch.min(distances, dim=1).values
        if PDE_testcases.FUNCTION_CASE == 11:
            sharp_min_distances = torch.where(x[:,0]**2 + x[:,1]**2 > 1.0, torch.tensor(-1.0, dtype=x.dtype, device=x.device), sharp_min_distances)
        return sharp_min_distances
class sharp_min_distance_function_hollig(torch.nn.Module):
    def __init__(self, model, delta = 0.1, gamma = 3.0):
        super().__init__()
        self.model = model
        self.delta = delta
        self.gamma = gamma

    def forward(self, x):
        distances = self.model(x)
        # Exact sharp minimum across the side distances.
        sharp_min_distances = torch.min(distances, dim=1).values
        # Apply Höllig transform to the scalar min distance (not the full side vector)
        sharp_min_distances = Geomertry.hollig_transform(d=sharp_min_distances, delta=self.delta, gamma=self.gamma)
        if PDE_testcases.FUNCTION_CASE == 11:
            sharp_min_distances = torch.where(x[:,0]**2 + x[:,1]**2 > 1.0, torch.tensor(-1.0, dtype=x.dtype, device=x.device), sharp_min_distances)
        return sharp_min_distances

class smooth_min_distance_function(torch.nn.Module):
    def __init__(self, model, alpha=10.0):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, x):
        distances = self.model(x)
        # Compute the smooth minimum distance function using log-sum-exp
        smooth_min_distances = -torch.logsumexp(-self.alpha * distances, dim=1) / self.alpha
        if PDE_testcases.FUNCTION_CASE == 11:
            smooth_min_distances = torch.where(x[:,0]**2 + x[:,1]**2 > 1.0, torch.tensor(-1.0, dtype=x.dtype, device=x.device), smooth_min_distances)
        return smooth_min_distances

class smooth_min_preserve_zero_distance_function_2side(torch.nn.Module):
    def __init__(self, model, k):
        super().__init__()
        self.model = model
        self.k = k # smoothness radius

    def forward(self, x):
        distances = self.model(x)
        sorted = torch.sort(distances,dim=1).values
        smallest = sorted[:,0]
        second = sorted[:,1]
        blended = (smallest*second)/((smallest+second))
        d = torch.where(smallest<0, -1, blended)
        d = d**self.k
        #d = blended
        if PDE_testcases.FUNCTION_CASE == 11:
            d = torch.where(x[:,0]**2 + x[:,1]**2 > 1.0, torch.tensor(-1.0, dtype=x.dtype, device=x.device), d)

        return d

class R_func_min(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        distances = self.model(x)
        # Compute the smooth minimum distance function using log-sum-exp
        num_sides = distances.shape[1]
        union_dist = distances[:,0]
        for i in range(1,num_sides):
            d1 = union_dist
            d2 = distances[:,i]
            union_dist = d1 + d2 - torch.sqrt(d1**2 + d2**2)

        if PDE_testcases.FUNCTION_CASE == 11:
            union_dist = torch.where(x[:,0]**2 + x[:,1]**2 > 1.0, torch.tensor(-1.0, dtype=x.dtype, device=x.device), union_dist)
        return union_dist

def get_s_param(model = model, numsides=4, point=None, track_grad=False):
    if track_grad:
        distances = model(point)
    else:
        with torch.no_grad():
            distances = model(point)
    assert distances.shape[1] == numsides
    s_coords = torch.zeros_like(distances)
    distances_ext = torch.cat([distances, distances[:, :2]], dim=1)
    #print(distances)
    for i in range(1,numsides+1):
        s_coords[:, i-1] = distances_ext[:, i-1] / (distances_ext[:, i-1] + distances_ext[:, i + 1])
    #print(s_coords)
    s_coords = torch.cat([s_coords[:, -1:], s_coords[:, :-1]], dim=1)
    #print(s_coords)
    return distances, s_coords


def _compute_lifting_values(d, bnd_values, smoothness=2):
    weights = 1.0 / (d ** smoothness)
    ud = torch.sum(bnd_values * weights, dim=1) / torch.sum(weights, dim=1)
    valid_mask = torch.all(d > 0, dim=1)
    return torch.where(valid_mask, ud, torch.zeros_like(ud))

def get_bnd_value(function_case, d, s):
    bnd_values = torch.zeros_like(d)
    if function_case == 11:
        vertices = Geomertry.regular_polygon_vertices_np(n_sides=5, radius=1, center=(0, 0))
        #get distancevector of each side

        #print("Vertices of the pentagon:", vertices)
        vertices_s = np.vstack([vertices, vertices[0]])
        diffs = vertices_s[1:,:] - vertices_s[:-1,:]
        diffs = torch.tensor(diffs, dtype=torch.float64)
        #print("Diffs:", diffs)
        #normalize diffs
        diffs = diffs #/ torch.norm(diffs, dim=1, keepdim=True)
        # get projection of points onto each side
        for i in range(d.shape[0]):
            s_ext = torch.cat([s[[i]], s[[i]]], dim=0)
            #print("s_ext:", s_ext)
            #print("diffs_norm:", diffs)
            shift = s_ext.T * diffs
            #print("shift:", shift)
            points = torch.tensor(vertices, dtype=torch.float64) + shift
            #print("points:", points)
            #print(model_MS(points))

            bnd_values[i,0]=0
            bnd_values[i,4]=0
            bnd_values[i,1]=1**2 - (points[1,0]**2 + points[1,1]**2)
            bnd_values[i,2]=1**2 - (points[2,0]**2 + points[2,1]**2)
            bnd_values[i,3]=1**2 - (points[3,0]**2 + points[3,1]**2)
        #print("Boundary values for function case 11:", bnd_values)

    else:
        raise NotImplementedError(f"Function case {function_case} not implemented.")
    return bnd_values


def get_lifting(model, function_case, num_sides, points, smoothness=2):
    # Implementation for getting lifting coordinates
    d,s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)
    ud = _compute_lifting_values(d=d, bnd_values=bnd_values, smoothness=smoothness)
    #ud = torch.zeros_like(ud)
    #print("WARNING: Liftin off")
    #print("Lifting values:", ud)
    return ud


def get_lifting_with_derivatives(model, function_case, num_sides, points, smoothness=2):
    """Return lifting values together with first and second derivatives.

    Returns
    -------
    ud : Tensor, shape (N,)
        Lifting values.
    dud : Tensor, shape (N, 2)
        First derivatives with respect to ``x`` and ``y``.
    hess_ud : Tensor, shape (N, 2, 2)
        Hessian matrix of the lifting function for each input point.
    """
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)

    d, s = get_s_param(model=model, numsides=num_sides, point=points, track_grad=True)
    bnd_values = get_bnd_value(function_case, d, s)
    ud = _compute_lifting_values(d=d, bnd_values=bnd_values, smoothness=smoothness)

    dud = torch.autograd.grad(
        outputs=ud.sum(),
        inputs=points,
        create_graph=True,
        retain_graph=True,
    )[0]

    hess_rows = []
    for axis_idx in range(points.shape[1]):
        second = torch.autograd.grad(
            outputs=dud[:, axis_idx].sum(),
            inputs=points,
            retain_graph=axis_idx < points.shape[1] - 1,
            allow_unused=False,
        )[0]
        hess_rows.append(second)

    hess_ud = torch.stack(hess_rows, dim=1)
    return ud, dud, hess_ud


def get_lifting_laplacian(model, function_case, num_sides, points, smoothness=2):
    """Return only the exact Laplacian of the lifting function.

    This is cheaper than ``get_lifting_with_derivatives`` because it avoids
    building the full Hessian tensor and computes only the trace of the
    Hessian, i.e. ``d2u/dx2 + d2u/dy2``.

    Returns
    -------
    ud : Tensor, shape (N,)
        Lifting values.
    lap_ud : Tensor, shape (N,)
        Exact Laplacian of the lifting function at the input points.
    """
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)

    d, s = get_s_param(model=model, numsides=num_sides, point=points, track_grad=True)
    bnd_values = get_bnd_value(function_case, d, s)
    ud = _compute_lifting_values(d=d, bnd_values=bnd_values, smoothness=smoothness)

    grad_ud = torch.autograd.grad(
        outputs=ud.sum(),
        inputs=points,
        create_graph=True,
        retain_graph=True,
    )[0]

    lap_ud = torch.zeros_like(ud)
    for axis_idx in range(points.shape[1]):
        second_axis = torch.autograd.grad(
            outputs=grad_ud[:, axis_idx].sum(),
            inputs=points,
            retain_graph=axis_idx < points.shape[1] - 1,
        )[0][:, axis_idx]
        lap_ud = lap_ud + second_axis

    return ud, lap_ud


def get_lifting_polygon(model, function_case, num_sides, points, smoothness=2, blend_method="wachspress", eps=1e-12):
    """Polygon-aware lifting for convex domains.

    This is an additional lifting helper that keeps the original
    ``get_lifting`` unchanged and adds a separate blending strategy.

    Supported blend methods:
    - ``inverse_distance``: original side-distance weighted average.
    - ``wachspress``: extended polygon blend using vertex weights built from
      adjacent side distances, with corner values reconstructed from adjacent
      side boundary values.
    - ``coons``: extended Coons-style blend that first interpolates along the
      polygon edges between corner values and then blends the edge patches by
      side proximity.
    """
    d, s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)

    ud = torch.zeros(points.shape[0], dtype=d.dtype, device=d.device)
    valid_mask = torch.all(d > 0, dim=1)
    if not torch.any(valid_mask):
        return ud

    ds = d[valid_mask]
    bs = bnd_values[valid_mask]
    ss = s[valid_mask]

    if blend_method == "inverse_distance":
        ud_vals = torch.sum((bs / (ds ** smoothness + eps)), dim=1) / torch.sum((1.0 / (ds ** smoothness + eps)), dim=1)
    else:
        corner_values = []
        for corner_idx in range(num_sides):
            left_side = (corner_idx - 1) % num_sides
            right_side = corner_idx % num_sides

            left_bnd = bs[:, left_side]
            right_bnd = bs[:, right_side]
            left_dist = ds[:, left_side]
            right_dist = ds[:, right_side]

            corner_value = (left_bnd * right_dist + right_bnd * left_dist) / (left_dist + right_dist + eps)
            corner_values.append(corner_value)

        corner_values = torch.stack(corner_values, dim=1)

        if blend_method == "wachspress":
            vertex_weights = []
            for corner_idx in range(num_sides):
                left_side = (corner_idx - 1) % num_sides
                right_side = corner_idx % num_sides
                weight = 1.0 / ((ds[:, left_side] + eps) * (ds[:, right_side] + eps))
                vertex_weights.append(weight)

            vertex_weights = torch.stack(vertex_weights, dim=1)
            vertex_weights = vertex_weights / (torch.sum(vertex_weights, dim=1, keepdim=True) + eps)
            ud_vals = torch.sum(vertex_weights * corner_values, dim=1)

        elif blend_method == "coons":
            edge_values = []
            for side_idx in range(num_sides):
                left_corner = side_idx
                right_corner = (side_idx + 1) % num_sides
                t = ss[:, side_idx]
                edge_values.append((1.0 - t) * corner_values[:, left_corner] + t * corner_values[:, right_corner])

            edge_values = torch.stack(edge_values, dim=1)
            side_weights = 1.0 / (ds ** smoothness + eps)
            side_weights = side_weights / (torch.sum(side_weights, dim=1, keepdim=True) + eps)
            ud_vals = torch.sum(side_weights * edge_values, dim=1)

        else:
            raise ValueError(f"Unknown blend_method '{blend_method}'. Use 'inverse_distance', 'wachspress', or 'coons'.")

    ud[valid_mask] = ud_vals
    ud[torch.any(d <= 0, dim=1)] = 0
    return ud

def get_lifting_R(model, function_case, num_sides, points, smoothness=1):
    # Build boundary selectors F_i directly from the implicit side distances.
    # For each side i, F_i is the R-conjunction of all other side functions.
    d, s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)

    def r_conjunction(a, b):
        return a + b - torch.sqrt(a ** 2 + b ** 2)

    selectors = []
    for omit_idx in range(num_sides):
        fi = None
        for side_idx in range(num_sides):
            if side_idx == omit_idx:
                continue
            fi = d[:, side_idx] if fi is None else r_conjunction(fi, d[:, side_idx])
        selectors.append(fi)

    fi = torch.stack(selectors, dim=1)
    fi_sum = torch.sum(fi, dim=1, keepdim=True)
    lambdas = torch.where(fi_sum > 0, fi / fi_sum, torch.zeros_like(fi))

    ud = torch.sum(lambdas * bnd_values, dim=1)
    ud[torch.any(d <= 0, dim=1)] = 0
    return ud


def get_lifting_tfi(model, function_case, num_sides, points, smoothness=1, corner_strength=2.0, eps=1e-12):
    """Compute lifting using a generalized transfinite interpolation (TFI)
    with corner correction.

    This method blends side boundary values using inverse-distance-type
    weights and adds vertex (corner) contributions computed from adjacent
    side values. The corner contributions are weighted by the product of
    adjacent side weights and a tunable `corner_strength`.

    Parameters
    - model, function_case, num_sides, points: same conventions as
      ``get_lifting``.
    - smoothness: exponent for inverse-distance weighting (>=0).
    - corner_strength: multiplicative weight applied to corner terms.
    - eps: small regularizer to avoid division by zero.

    Returns
    - ud: Tensor of lifting values with shape (N,)
    """
    d, s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)

    # result tensor
    ud = torch.zeros(points.shape[0], dtype=d.dtype, device=d.device)

    # Only compute for points where all side distances are positive
    valid_mask = torch.all(d > 0, dim=1)
    if not torch.any(valid_mask):
        return ud

    ds = d[valid_mask]
    bs = bnd_values[valid_mask]

    # inverse-distance weights (similar spirit to get_lifting)
    side_w = 1.0 / (ds ** smoothness + eps)
    side_w_sum = torch.sum(side_w, dim=1, keepdim=True)
    side_w = side_w / (side_w_sum + eps)

    # side contribution
    side_num = torch.sum(side_w * bs, dim=1)

    # build vertex values as averages of adjacent side boundary values
    n = num_sides
    # bs: (M, n)
    v_vals = []
    for i in range(n):
        j = (i + 1) % n
        v_vals.append(0.5 * (bs[:, i] + bs[:, j]))
    v_vals = torch.stack(v_vals, dim=1)  # (M, n) vertex contributions indexed by vertex i

    # corner weights: product of adjacent normalized side weights
    corner_w = []
    for i in range(n):
        j = (i + 1) % n
        corner_w.append(side_w[:, i] * side_w[:, j])
    corner_w = torch.stack(corner_w, dim=1)  # (M, n)

    corner_num = torch.sum(corner_w * v_vals, dim=1)
    corner_den = torch.sum(corner_w, dim=1)

    num = side_num + corner_strength * corner_num
    den = torch.sum(side_w, dim=1) + corner_strength * corner_den + eps

    ud_vals = num / den

    ud[valid_mask] = ud_vals
    # where any of d is negative set ud to 0 (outside/intersecting cases)
    ud[torch.any(d <= 0, dim=1)] = 0
    return ud


def get_lifting_corner(model, function_case, num_sides, points, smoothness=1, eps=1e-12, closest_corner_only=False):
    # Corner-based lifting: interpolate the two sides meeting at each corner,
    # then blend the corner values using distance-to-corner weights.
    d, s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)

    if function_case == 11:
        vertices = torch.tensor(
            Geomertry.regular_polygon_vertices_np(n_sides=num_sides, radius=1, center=(0, 0)),
            dtype=torch.float64,
            device=points.device,
        )
    else:
        raise NotImplementedError(f"Function case {function_case} not implemented.")

    ud = torch.zeros(points.shape[0], dtype=d.dtype, device=d.device)

    valid_mask = torch.all(d > 0, dim=1)
    if not torch.any(valid_mask):
        return ud

    ds = d[valid_mask]
    bs = bnd_values[valid_mask]
    pts = points[valid_mask]

    #apply hollig transform to the distance values
    #ds = Geomertry.hollig_transform(d=ds, delta=0.05, gamma=1.0)
    
    # Build one corner value per vertex.
    # Corner i is formed by side i-1 and side i (cyclic indexing).
    corner_values = []
    for corner_idx in range(num_sides):
        left_side = (corner_idx - 1) % num_sides
        right_side = corner_idx % num_sides

        left_bnd = bs[:, left_side]
        right_bnd = bs[:, right_side]
        left_dist = ds[:, left_side]
        right_dist = ds[:, right_side]

        # Inverse-distance interpolation between the two boundary values.
        corner_value = (left_bnd * right_dist + right_bnd * left_dist) / (left_dist + right_dist + eps)
        corner_values.append(corner_value)

    corner_values = torch.stack(corner_values, dim=1)

    # Blend the corner contributions by distance to the corresponding vertex.
    corner_distances = torch.sqrt(torch.sum((pts[:, None, :] - vertices[None, :, :]) ** 2, dim=2) + eps)
    # Old blending rule kept for reference:
    corner_weights = 1.0 / (corner_distances ** smoothness + eps)
    corner_weights = corner_weights / (torch.sum(corner_weights, dim=1, keepdim=True) + eps)

    # Alternative blending: softmax over negative corner distances.
    # This gives a smoother preference for the closest corner without
    # letting distant corners dominate as strongly as inverse-distance weights.
    #corner_scale = torch.clamp(torch.as_tensor(smoothness, dtype=d.dtype, device=d.device), min=eps)
    #corner_weights = torch.softmax(-corner_distances / corner_scale, dim=1)
    if closest_corner_only:
        closest_corner = torch.argmin(corner_distances, dim=1)
        corner_weights = torch.zeros_like(corner_weights)
        corner_weights.scatter_(1, closest_corner[:, None], 1.0)

    ud_vals = torch.sum(corner_weights * corner_values, dim=1)
    ud[valid_mask] = ud_vals
    ud[torch.any(d <= 0, dim=1)] = 0
    return ud

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
    extent = None,
    cmap: str = 'viridis',
    show: bool = True,
):
    """Plot the lifting function u_d computed by get_lifting over the domain.

    Parameters:
    - model: neural distance model used by get_s_param/get_lifting
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

    # Compute lifting (returns torch tensor)
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


def plot_s_parameter_isocurves(
    model=model,
    function_case: int = 11,
    num_sides: int = 5,
    N: int = 200,
    extent=None,
    levels=(0.25, 0.5, 0.75),
    cmap: str = 'viridis',
    show: bool = True,
):
    """Plot isocurves of the s-parameters, one subplot per side.

    The function evaluates the distance outputs and the associated s-coordinates
    returned by ``get_s_param`` on a uniform grid, then draws contour lines for
    each side parameter s_i.

    Parameters
    ----------
    model : torch.nn.Module
        Distance model used by ``get_s_param``.
    function_case : int
        Geometry/test case used to choose the plotting window.
    num_sides : int
        Number of side parameters to visualize.
    N : int
        Grid resolution per axis.
    extent : tuple or None
        Optional ``(xmin, xmax, ymin, ymax)`` plotting window.
    levels : sequence of float
        Contour levels to show for each s-parameter.
    cmap : str
        Colormap used for the background scalar field.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    X, Y, S : ndarray
        Meshgrid arrays and s-parameter values with shape ``(num_sides, N, N)``.
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
        distances, s_coords = get_s_param(model=model, numsides=num_sides, point=pts_t)

    S = s_coords.detach().cpu().numpy().reshape(N, N, num_sides).transpose(2, 0, 1)
    D = distances.detach().cpu().numpy().reshape(N, N, num_sides).transpose(2, 0, 1)

    nrows = int(np.ceil(num_sides / 2))
    ncols = 2 if num_sides > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for side_idx in range(num_sides):
        ax = axes[side_idx]
        background = ax.contourf(X, Y, S[side_idx], levels=60, cmap=cmap)
        contour = ax.contour(X, Y, S[side_idx], levels=list(levels), colors='white', linewidths=1.0)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%0.2f')
        ax.set_title(f's-parameter isocurves, side {side_idx + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(background, ax=ax, fraction=0.046, pad=0.04)

        # Overlay the zero level-set of the signed distance model if available.
        try:
            ax.contour(X, Y, np.min(D, axis=0), levels=[0.0], colors='red', linewidths=1.5)
        except Exception:
            pass

    for ax in axes[num_sides:]:
        ax.axis('off')

    fig.suptitle('Isocurves of the s-parameters', y=1.02)

    if show:
        plt.show()

    return X, Y, S



if __name__ == "__main__":
    #SDF.plotDisctancefunction(eval_fun=model,N=100,extent=(-1.1, 1.1, -1.1, 1.1), contour=True)
    #get_s_param(model=model, numsides=5, point=torch.tensor([[0.4, 0.0],[0.1,0.1]]))
    
    
    #model_MS = netdefs.load_test_model("SIREN_pentagon_MSSDF_02", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})
    model_MS = Geomertry.AnaliticalDistancePentagon_SideDistances()
    model = sharp_min_distance_function(model_MS)
    #model = smooth_min_distance_function(model_MS, alpha=20.0)
    #model = smooth_min_distance_function_preserve_zero(model=model_MS, alpha=5)
    #model = R_func_min(model=model_MS)
    get_lifting(model_MS, function_case=11, num_sides=5, points=torch.tensor([[-0.80901699, 0.0],[0.0,-0.80901699],[0,0]]))

    #plot the distance funcion over [-1,1]x[-1,1]
    X = np.linspace(-1.1, 1.1, 100)
    Y = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(X, Y)
    points = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float64)
    distances = model(points).detach().numpy().reshape(X.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, distances, levels=50, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.contour(X, Y, distances, levels=[0], colors='red', linewidths=2)
    plt.title('Smooth Minimum Distance Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    #plot_s_parameter_isocurves(model=model_MS, function_case=11, num_sides=5, N=200, extent=(-1, 1, -1, 1), levels=(0.1, 0.25, 0.5, 0.75, 0.9), cmap='viridis', show=True)
    plot_lifting_function(model=model_MS, function_case=11, num_sides=5, N=200, extent=(-1, 1, -1, 1), cmap='viridis', show=True)
    