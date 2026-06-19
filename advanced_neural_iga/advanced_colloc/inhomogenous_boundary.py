import collocation_WEB as cWEB
import numpy as np
import matplotlib.pyplot as plt
import importlib
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
        # Compute the sharp minimum distance function
        sharp_min_distances = torch.min(distances, dim=1).values
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
        return smooth_min_distances
class smooth_min_preserve_zero_distance_function(torch.nn.Module):
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
        #d = blended

        return d

def get_s_param(model = model, numsides=4, point=None):
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
        diffs = diffs / torch.norm(diffs, dim=1, keepdim=True)
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
        #! ide a projekció kell TODO- ez most jó vagy sem??
    else:
        raise NotImplementedError(f"Function case {function_case} not implemented.")
    return bnd_values

def get_lifting(model, function_case, num_sides, points):
    # Implementation for getting lifting coordinates
    d,s = get_s_param(model=model, numsides=num_sides, point=points)
    bnd_values = get_bnd_value(function_case, d, s)
    ud = torch.sum((bnd_values/d), dim=1)/torch.sum((1/d), dim=1)
    # where any of d is negative set ud to 0
    ud[torch.any(d <= 0, dim=1)] = 0

    print("Lifting values:", ud)
    return ud


def plot_case_11_poisson_samples(
    recon_info,
    model,
    csv_path=None,
    title_prefix="Function case 11",
    show: bool = True,
    filename: str = "poisson_samples_hom_1_e_3.csv",
    model_ms=None,
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
    pred = cWEB.reconstruct_collocation_at_points(pts_x, pts_y, recon_info, wfct_phys, model_ms=model_ms)
    error = pred - gt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    plots = [
        (gt, f"{title_prefix} ground truth", "viridis"),
        (pred, f"{title_prefix} reconstructed solution", "viridis"),
        (np.abs(error), f"{title_prefix} absolute error", "hot"),
    ]

    for ax, (values, title, cmap) in zip(axes, plots):
        scatter = ax.scatter(pts_x, pts_y, c=values, s=36, cmap=cmap, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

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
    
    
    model_MS = netdefs.load_test_model("SIREN_pentagon_MSSDF_large", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})
    #model = sharp_min_distance_function(model_MS)
    #model = smooth_min_distance_function(model_MS, alpha=10.0)
    model = smooth_min_preserve_zero_distance_function(model=model_MS, k = 1)
    get_lifting(model_MS, function_case=11, num_sides=5, points=torch.tensor([[0.4, 0.0],[0.1,0.1],[0,0]]))

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
    plot_s_parameter_isocurves(model=model_MS, function_case=11, num_sides=5, N=200, extent=(-1, 1, -1, 1), levels=(0.1, 0.25, 0.5, 0.75, 0.9), cmap='viridis', show=True)
    plot_lifting_function(model=model_MS, function_case=11, num_sides=5, N=200, extent=(-1, 1, -1, 1), cmap='viridis', show=True)
    