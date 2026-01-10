import torch
import numpy as np
import matplotlib.pyplot as plt
import geometry_bspline as bsp_geom
import geometry_definitions as geom_defs
def plot_normal_vectors_on_bspline(control_points, degree=3, num_vectors=20, vector_length=0.2, device=None):
    """
    Plot normal vectors on a B-spline curve.
    
    Args:
        control_points: tensor of shape (num_cp, 2) - B-spline control points
        degree: int - B-spline degree
        num_vectors: int - number of normal vectors to plot
        vector_length: float - length of normal vectors
        device: torch device
    """
    import matplotlib.pyplot as plt
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    control_points = control_points.to(device)
    knots = bsp_geom.create_knot_vector(control_points.shape[0], degree, closed=True).to(device)
    
    # Sample parameter values for normal vectors
    t_values = torch.linspace(knots[degree], knots[-degree-1], num_vectors, device=device)
    
    # Evaluate curve points
    curve_points = bsp_geom.evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device)
    
    # Compute normal vectors
    normals = bsp_geom.bspline_normalvectors(t_values, control_points, knots, degree, device)
    
    # Scale normals
    normals_scaled = normals * vector_length
    
    # Plotting
    curve_points_cpu = curve_points.cpu().numpy()
    normals_cpu = normals_scaled.cpu().numpy()
    control_points_cpu = control_points.cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.plot(control_points_cpu[:, 0], control_points_cpu[:, 1], 'ro--', label='Control Points', markersize=5)
    plt.plot(curve_points_cpu[:, 0], curve_points_cpu[:, 1], 'b-', label='B-spline Curve', linewidth=2)
    
    for i in range(num_vectors):
        start = curve_points_cpu[i]
        end = start + normals_cpu[i]
        plt.arrow(start[0], start[1], normals_cpu[i, 0], normals_cpu[i, 1],
                  head_width=0.02, head_length=0.04, fc='g', ec='g')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'B-spline Curve with Normal Vectors (degree={degree})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_model_error_map(model, ctrl_pts, degree=1, N=200, extent=(-1.1, 1.1, -1.1, 1.1), use_log=False, device=None, num_curve_samples=1000):
    """
    Plot error map of a neural implicit model against B-spline signed distance function.
    
    Args:
        model: neural implicit model with callable interface
        ctrl_pts: tensor of shape (num_cp, 2) - B-spline control points
        N: int - resolution per axis
        extent: tuple - (xmin, xmax, ymin, ymax) plotting extent
        device: torch device
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ctrl_pts = ctrl_pts.to(device)
    
    # Create grid
    x_vals = np.linspace(extent[0], extent[1], N)
    y_vals = np.linspace(extent[2], extent[3], N)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten grid points
    pts_np = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float64)
    pts = torch.from_numpy(pts_np).to(device)
    
    # Compute model predictions in chunks
    model_values = torch.empty(pts.shape[0], device=device, dtype=torch.float64)
    chunk_size = 200000
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            model_values[i:j] = model(pts[i:j]).squeeze()
    
    # Compute true signed distances in chunks
    true_distances = torch.empty(pts.shape[0], device=device, dtype=torch.float64)
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            true_distances[i:j] = bsp_geom.bspline_signed_distance_vectorized(
                pts[i:j], ctrl_pts, degree=degree, device=device, num_curve_samples=num_curve_samples
            )
    plt.figure(figsize=(8, 8))
    # Compute absolute error
    errors = torch.abs(model_values - true_distances)
    Z = errors.cpu().numpy().reshape(N, N)
    if not use_log:
        plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    else:
        plt.contourf(X, Y, Z, levels=50, locator=ticker.LogLocator(), cmap='plasma')
    
    
    plt.colorbar(label='Absolute Error')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Error Map')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.grid(True)
    plt.show()

def plot_bspline_curve(control_points, degree=3, num_samples=400, closed=True, device=None,
                        show_control_points=True, curve_color='C0', control_color='C3', linewidth=2):
    """
    Plot a B-spline curve given control points and a uniform knot vector.

    Args:
        control_points: tensor (num_cp,2) or numpy array of control points
        degree: spline degree (int)
        num_samples: number of parameter samples on [0,1]
        closed: if True uses periodic/uniform knot vector for closed curve, otherwise uses open-uniform (clamped)
        device: torch device (defaults to cuda if available)
        show_control_points: plot control polygon if True
        curve_color: matplotlib color for the curve
        control_color: matplotlib color for control polygon
        linewidth: line width for the curve
    Returns:
        curve_points (torch.Tensor) of shape (num_samples, 2)
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert control points to torch tensor on the chosen device
    if isinstance(control_points, np.ndarray):
        control_points = torch.from_numpy(control_points.astype(np.float64))
    control_points = control_points.to(device).double()

    # Create a uniform knot vector (uses existing helper)
    knots = bsp_geom.create_knot_vector(control_points.shape[0], degree, closed=closed).to(device)

    # Parameter samples
    t = torch.linspace(knots[degree], knots[-degree-1], num_samples, device=device)

    # Evaluate curve
    curve_points = bsp_geom.evaluate_bspline_curve_vectorized(t, control_points, knots, degree, device=device)

    # Convert to numpy for plotting
    cp = control_points.cpu().numpy()
    curve_np = curve_points.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(curve_np[:, 0], curve_np[:, 1], color=curve_color, linewidth=linewidth, label='B-spline curve')
    if show_control_points:
        plt.plot(cp[:, 0], cp[:, 1], 'o--', color=control_color, label='Control polygon', markersize=5)
        # If closed, show closing segment
        if closed:
            plt.plot([cp[-1, 0], cp[0, 0]], [cp[-1, 1], cp[0, 1]], '--', color=control_color, alpha=0.6)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'B-spline curve (degree={degree}, control pts={cp.shape[0]})')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_bspline_distance_field(control_points, degree=3, N=400, extent=(-2, 2, -2, 2), 
                               contour=False, levels=20, device=None, chunk_size=200000, use_refinement=False):
    """
    Plot the signed distance field for a B-spline contour.
    
    Args:
        control_points: tensor of shape (num_cp, 2) - B-spline control points
        degree: int - B-spline degree
        N: int - resolution per axis
        extent: tuple - (xmin, xmax, ymin, ymax) plotting extent
        contour: bool - use contour lines instead of filled contours
        levels: int - number of contour levels
        device: torch device
        chunk_size: int - process points in chunks to manage memory
    """
    import matplotlib.pyplot as plt
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    control_points = control_points.to(device)
    
    # Create grid
    x_vals = np.linspace(extent[0], extent[1], N)
    y_vals = np.linspace(extent[2], extent[3], N)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten grid points
    pts_np = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float64)
    pts = torch.from_numpy(pts_np).to(device)
    
    # Compute distances in chunks
    distances = torch.empty(pts.shape[0], device=device, dtype=torch.float64)
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            distances[i:j] = bsp_geom.bspline_signed_distance_vectorized(
                pts[i:j], control_points, degree=degree, device=device, use_refinement=use_refinement
            )
    
    Z = distances.cpu().numpy().reshape(N, N)
    
    plt.figure(figsize=(8, 8))
    if contour:
        cs = plt.contour(X, Y, Z, levels=levels)
        plt.clabel(cs, inline=True, fontsize=8)
    else:
        plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
        plt.colorbar(label='Signed Distance')
    
    # Plot zero level set (the contour itself)
    plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
    
    # Plot control points
    cp_cpu = control_points.cpu().numpy()
    plt.plot(cp_cpu[:, 0], cp_cpu[:, 1], 'ro-', markersize=6, linewidth=1, 
             alpha=0.7, label='Control Points')
    
    #points_on_curve = generate_points_on_curve(control_points, degree=degree, num_points=100, device=device)
    #points_on_curve_cpu = points_on_curve.cpu().numpy()
    #plt.plot(points_on_curve_cpu[:, 0], points_on_curve_cpu[:, 1], 'go', markersize=8, label='Curve Sample Points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'B-spline Signed Distance Field (degree={degree})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_laplacian_of_bspline_sdf(model = None, control_points=None, degree=1, num_samples=1000):
    """
    Plot the Laplacian of the signed distance function defined by a B-spline curve.
    
    Args:
        model: neural implicit model with callable interface
    """
    N = 200                         # grid resolution
    x_vals = np.linspace(-1, 1, N)
    y_vals = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float64)

    with torch.no_grad():
        if model is None:
            if control_points is None:
                #default rounded star shape
                control_points = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=2)
                degree = 2
            Z = bsp_geom.bspline_signed_distance_vectorized(pts_t, control_points, degree=degree,num_curve_samples=num_samples).numpy().reshape(N, N)
        else:
            Z = model(pts_t).cpu().numpy().reshape(N, N)
    # plot laplace Z
    from scipy.ndimage import laplace
    Z_laplace = laplace(Z)
    Z_abs = np.abs(Z_laplace)
    plt.figure(figsize=(10, 10))
    plt.title('Laplacian of the SDF')
    plt.axis('equal')
    plt.contourf(X, Y, Z_abs, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.show()