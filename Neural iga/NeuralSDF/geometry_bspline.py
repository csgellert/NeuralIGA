import torch
import numpy as np
import matplotlib.pyplot as plt
def bspline_basis_functions(t, knots, degree, device=None):
    """
    Compute B-spline basis functions using Cox-de Boor recursion formula.
    Vectorized implementation for multiple parameter values.
    
    Args:
        t: tensor of shape (num_t,) - parameter values to evaluate at
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline
        device: torch device
    
    Returns:
        tensor of shape (num_t, num_basis) - basis function values
    """
    if device is None:
        device = t.device
    
    t = t.to(device)
    knots = knots.to(device)
    
    num_t = t.shape[0]
    num_knots = knots.shape[0]
    num_basis = num_knots - degree - 1
    
    # Initialize basis functions
    basis = torch.zeros(num_t, num_basis, device=device)
    
    # Degree 0 (piecewise constant)
    for i in range(num_basis):
        mask = (t >= knots[i]) & (t < knots[i + 1])
        # Handle the last interval boundary
        if i == num_basis - 1:
            mask = mask | (t == knots[i + 1])
        basis[:, i] = mask.float()
    
    # Higher degrees using Cox-de Boor recursion
    for d in range(1, degree + 1):
        new_basis = torch.zeros(num_t, num_basis, device=device)
        for i in range(num_basis):
            # Left term
            if i + d < num_knots and knots[i + d] != knots[i]:
                left_coeff = (t - knots[i]) / (knots[i + d] - knots[i])
                new_basis[:, i] += left_coeff * basis[:, i]
            
            # Right term
            if i + 1 < num_basis and i + d + 1 < num_knots and knots[i + d + 1] != knots[i + 1]:
                right_coeff = (knots[i + d + 1] - t) / (knots[i + d + 1] - knots[i + 1])
                new_basis[:, i] += right_coeff * basis[:, i + 1]
        
        basis = new_basis
    
    return basis

def bspline_basis_derivatives(t, knots, degree, device=None):
    """
    Compute derivatives of B-spline basis functions using recursive formula.
    Vectorized implementation for multiple parameter values.
    
    Args:
        t: tensor of shape (num_t,) - parameter values to evaluate at
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline
        device: torch device
    """
    if device is None:
        device = t.device
    
    t = t.to(device)
    knots = knots.to(device)
    
    num_t = t.shape[0]
    num_knots = knots.shape[0]
    num_basis = num_knots - degree - 1
    
    # Compute basis functions of degree-1
    basis_lower = bspline_basis_functions(t, knots, degree - 1, device)
    
    # Initialize derivative basis functions
    deriv_basis = torch.zeros(num_t, num_basis, device=device)
    
    for i in range(num_basis):
        # Left term
        if knots[i + degree] != knots[i]:
            left_coeff = degree / (knots[i + degree] - knots[i])
            deriv_basis[:, i] += left_coeff * basis_lower[:, i]
        
        # Right term
        if i + 1 < num_basis and knots[i + degree + 1] != knots[i + 1]:
            right_coeff = degree / (knots[i + degree + 1] - knots[i + 1])
            deriv_basis[:, i] -= right_coeff * basis_lower[:, i + 1]
    
    return deriv_basis

def bspline_normalvectors(t, control_points, knots, degree, device=None):
    """
    Compute normal vectors of B-spline curve at parameter values t.
    
    Args:
        t: tensor of shape (num_t,) - parameter values [0, 1]
        control_points: tensor of shape (num_cp, 2) - control points [x, y]
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline
        device: torch device
    """
    
    if device is None:
        device = control_points.device
    
    t = t.to(device)
    control_points = control_points.to(device)
    knots = knots.to(device)
    
    # Compute basis function derivatives
    deriv_basis = bspline_basis_derivatives(t, knots, degree, device)  # (num_t, num_basis)
    
    # Compute tangent vectors
    tangents = torch.matmul(deriv_basis, control_points)  # (num_t, 2)
    
    # Compute normal vectors by rotating tangents 90 degrees
    normals = torch.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    
    # Normalize normal vectors
    norms = torch.norm(normals, dim=1, keepdim=True) + 1e-8  # Avoid division by zero
    normals = normals / norms
    
    return normals

def evaluate_bspline_curve_vectorized(t, control_points, knots, degree, device=None):
    """
    Evaluate B-spline curve at multiple parameter values.
    
    Args:
        t: tensor of shape (num_t,) - parameter values [0, 1]
        control_points: tensor of shape (num_cp, 2) - control points [x, y]
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline
        device: torch device
    
    Returns:
        tensor of shape (num_t, 2) - curve points [x, y]
    """
    if device is None:
        device = control_points.device
    
    t = t.to(device)
    control_points = control_points.to(device)
    knots = knots.to(device)
    
    # Compute basis functions
    basis = bspline_basis_functions(t, knots, degree, device)  # (num_t, num_basis)
    #import bspline 
    #knots = knots.cpu().numpy()
    #t = t.cpu().numpy()
    #basis = bspline.Bspline(knots, degree).collmat(t)
    # Convert basis back to torch tensor
    #basis = torch.from_numpy(basis).to(device).float()
    # Evaluate curve points
    curve_points = torch.matmul(basis, control_points)  # (num_t, 2)
    
    return curve_points

def create_knot_vector(num_control_points, degree, closed=True):
    if closed:
        # For closed curves, use periodic knot vector
        num_knots = num_control_points + degree + 1
        knots = torch.arange(num_knots, dtype=torch.float32)
        knots = knots / (num_knots - 1)  # Normalize to [0, 1] range
    else:
        # For open curves, clamp knots at endpoints
        num_knots = num_control_points + degree + 1
        knots = torch.zeros(num_knots, dtype=torch.float32)
        
        # Interior knots
        if num_knots > 2 * (degree + 1):
            interior_knots = torch.linspace(0, 1, num_knots - 2 * (degree + 1) + 2)[1:-1]
            knots[degree + 1:num_knots - degree - 1] = interior_knots
        
        # Clamp end knots
        knots[degree + 1:] = 1.0
    
    return knots

def distance_points_to_bspline_curve_vectorized(query_points, control_points, degree=3, 
                                              num_curve_samples=1000, device=None):
    """
    Compute distances from query points to a B-spline curve.
    
    Args:
        query_points: tensor of shape (num_points, 2) - points to compute distances for
        control_points: tensor of shape (num_cp, 2) - B-spline control points
        degree: int - degree of B-spline (default 3 for cubic)
        num_curve_samples: int - number of samples to discretize curve
        device: torch device
    
    Returns:
        tensor of shape (num_points,) - minimum distances to curve
    """
    if device is None:
        device = query_points.device
    
    query_points = query_points.to(device)
    control_points = control_points.to(device)
    
    # Create knot vector for closed curve
    knots = create_knot_vector(control_points.shape[0], degree, closed=True)
    knots = knots.to(device)
    
    # Sample curve points
    t_values = torch.linspace(knots[degree], knots[-degree-1], num_curve_samples, device=device)
    curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device)
    
    # Compute distances from query points to all curve points
    # query_points: (num_points, 2), curve_points: (num_samples, 2)
    # Expand dimensions for broadcasting
    query_expanded = query_points.unsqueeze(1)  # (num_points, 1, 2)
    curve_expanded = curve_points.unsqueeze(0)   # (1, num_samples, 2)
    
    # Compute squared distances
    diff = query_expanded - curve_expanded  # (num_points, num_samples, 2)
    distances_sq = torch.sum(diff ** 2, dim=2)  # (num_points, num_samples)
    
    # Find minimum distance for each query point
    min_distances = torch.sqrt(torch.min(distances_sq, dim=1)[0])  # (num_points,)
    
    return min_distances

def point_in_polygon_winding_number(points, polygon_vertices):
    """
    Determine if points are inside polygon using winding number algorithm.
    This reference implementation uses Python loops and is kept for backwards compatibility.
    Prefer point_in_polygon_winding_number_batched for performance.
    """
    num_points = points.shape[0]
    num_vertices = polygon_vertices.shape[0]

    winding_numbers = torch.zeros(num_points, device=points.device)

    for i in range(num_vertices):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % num_vertices]

        upward = (v1[1] <= points[:, 1]) & (points[:, 1] < v2[1])
        downward = (v2[1] <= points[:, 1]) & (points[:, 1] < v1[1])

        for j in range(num_points):
            if upward[j]:
                cross = (v2[0] - v1[0]) * (points[j, 1] - v1[1]) - (v2[1] - v1[1]) * (points[j, 0] - v1[0])
                if cross > 0:
                    winding_numbers[j] += 1
            elif downward[j]:
                cross = (v2[0] - v1[0]) * (points[j, 1] - v1[1]) - (v2[1] - v1[1]) * (points[j, 0] - v1[0])
                if cross < 0:
                    winding_numbers[j] -= 1

    return winding_numbers != 0

def point_in_polygon_winding_number_batched(points, polygon_vertices, batch_size=200000):
    """
    Vectorized and batched winding number test.

    Args:
        points: (N,2) tensor
        polygon_vertices: (M,2) tensor defining a closed polygon (last edge wraps to first)
        batch_size: number of points to process per batch to control memory

    Returns:
        inside_mask: (N,) boolean tensor
    """
    device = points.device
    dtype = points.dtype

    v1 = polygon_vertices
    v2 = torch.roll(polygon_vertices, shifts=-1, dims=0)

    v1x = v1[:, 0].to(dtype)
    v1y = v1[:, 1].to(dtype)
    v2x = v2[:, 0].to(dtype)
    v2y = v2[:, 1].to(dtype)

    dvx = (v2x - v1x)
    dvy = (v2y - v1y)

    E = v1.shape[0]
    N = points.shape[0]

    inside = torch.empty(N, dtype=torch.bool, device=device)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        pb = points[i:j].to(device, dtype=dtype)
        px = pb[:, 0].unsqueeze(1)  # (B,1)
        py = pb[:, 1].unsqueeze(1)  # (B,1)

        v1y_row = v1y.unsqueeze(0)  # (1,E)
        v2y_row = v2y.unsqueeze(0)
        v1x_row = v1x.unsqueeze(0)

        upward = (v1y_row <= py) & (py < v2y_row)      # (B,E)
        downward = (v2y_row <= py) & (py < v1y_row)    # (B,E)

        # Cross product to determine sidedness for each point-edge pair
        cross = dvx.unsqueeze(0) * (py - v1y_row) - dvy.unsqueeze(0) * (px - v1x_row)  # (B,E)

        inc = (upward & (cross > 0)).to(torch.int32)
        dec = (downward & (cross < 0)).to(torch.int32)
        winding = (inc - dec).sum(dim=1)

        inside[i:j] = winding != 0

    return inside

def distance_points_to_curve_samples_batched(query_points, curve_points, batch_size=200000):
    """
    Compute minimal euclidean distance from each query point to a set of curve samples, in batches.

    Args:
        query_points: (N,2) tensor
        curve_points: (S,2) tensor
        batch_size: number of query points per batch

    Returns:
        min_distances: (N,) tensor of minimal distances
    """
    device = query_points.device
    dtype = query_points.dtype

    N = query_points.shape[0]
    min_d = torch.empty(N, device=device, dtype=dtype)

    # Ensure same dtype
    curve_points = curve_points.to(device=device, dtype=dtype)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        qb = query_points[i:j].to(device=device, dtype=dtype)
        # torch.cdist is often well-optimized on GPU and CPU
        dists = torch.cdist(qb, curve_points)  # (B, S)
        min_d[i:j], _ = torch.min(dists, dim=1)

    return min_d

def bspline_signed_distance_vectorized(query_points, control_points, degree=3,
                                     num_curve_samples=1000, num_polygon_samples=500,
                                     device=None, point_batch_size=200000,
                                     precomputed_curve_points=None,
                                     precomputed_polygon_vertices=None,
                                     precomputed_knots=None):
    """
    Compute signed distance from query points to a closed B-spline contour.
    Positive distances indicate points inside the contour, negative for outside.
    
    Args:
        query_points: tensor of shape (num_points, 2) or (2,) - points to evaluate
        control_points: tensor of shape (num_cp, 2) - B-spline control points defining closed contour
        degree: int - degree of B-spline (default 3 for cubic)
        num_curve_samples: int - samples for distance computation
        num_polygon_samples: int - samples for inside/outside determination
        device: torch device
    
    Returns:
        tensor of shape (num_points,) or scalar - signed distances
    """
    # Handle single point input
    if query_points.dim() == 1:
        query_points = query_points.unsqueeze(0)
        single_point = True
    else:
        single_point = False
    
    if device is None:
        device = query_points.device

    # Use float32 for speed/memory unless higher precision is explicitly provided
    dtype = torch.float32 if query_points.dtype == torch.float32 else query_points.dtype

    query_points = query_points.to(device=device, dtype=dtype)
    control_points = control_points.to(device=device, dtype=dtype)

    # Prepare knots
    if precomputed_knots is not None:
        knots = precomputed_knots.to(device=device, dtype=dtype)
    else:
        knots = create_knot_vector(control_points.shape[0], degree, closed=True).to(device=device, dtype=dtype)

    # Curve samples for distances
    if precomputed_curve_points is not None:
        curve_points = precomputed_curve_points.to(device=device, dtype=dtype)
    else:
        t_values = torch.linspace(knots[degree], knots[-degree-1], num_curve_samples, device=device, dtype=dtype)
        curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device)

    # Polygon vertices for inside test (can be coarser)
    if precomputed_polygon_vertices is not None:
        polygon_vertices = precomputed_polygon_vertices.to(device=device, dtype=dtype)
    else:
        t_polygon = torch.linspace(knots[degree], knots[-degree-1], num_polygon_samples, device=device, dtype=dtype)
        polygon_vertices = evaluate_bspline_curve_vectorized(t_polygon, control_points, knots, degree, device)

    # Compute minimum distances to the curve samples in batches
    min_distances = distance_points_to_curve_samples_batched(query_points, curve_points, batch_size=point_batch_size)

    # Batched winding number inside/outside test
    inside_mask = point_in_polygon_winding_number_batched(query_points, polygon_vertices, batch_size=point_batch_size)

    # Apply sign: positive if inside, negative if outside
    signed_distances = torch.where(inside_mask, min_distances, -min_distances)
    
    # Return scalar if input was single point
    if single_point:
        return signed_distances[0]
    
    return signed_distances

def create_circle_bspline_control_points(center=(0.0, 0.0), radius=1.0, num_points=8, degree=3, device=None):
    """
    Create control points for a circular B-spline curve.
    
    Args:
        center: tuple (x, y) - center of circle
        radius: float - radius of circle
        num_points: int - number of control points (should be multiple of 4 for good circle approximation)
        degree: int - degree of the B-spline
        device: torch device
    
    Returns:
        tensor of shape (num_points, 2) - control points
    """
    if device is None:
        device = torch.device('cpu')
    
    # Create control points in a circle
    angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]  # Exclude duplicate
    
    # For better circle approximation with B-splines, use slightly larger radius for control points
    # This compensates for the fact that B-splines don't pass through all control points
    control_radius = radius * 1.1  # Adjustment factor for better circle approximation
    
    x = center[0] + control_radius * torch.cos(angles)
    y = center[1] + control_radius * torch.sin(angles)
    # periodic extension
    x = torch.cat([x, x[:degree]])
    y = torch.cat([y, y[:degree]])
    control_points = torch.stack([x, y], dim=1)
    
    return control_points

def create_star_bspline_control_points(center=(0.0, 0.0), outer_radius=1.0, inner_radius=0.5, 
                                     num_star_points=5, degree=3, device=None):
    """
    Create control points for a star-shaped B-spline curve.
    
    Args:
        center: tuple (x, y) - center of star
        outer_radius: float - radius to outer star points
        inner_radius: float - radius to inner star points  
        num_star_points: int - number of star points
        device: torch device
    
    Returns:
        tensor of shape (num_control_points, 2) - control points
    """
    if device is None:
        device = torch.device('cpu')
    
    num_control_points = num_star_points * 2
    control_points = torch.zeros(num_control_points, 2, device=device)
    
    angle_step = 2 * torch.pi / num_control_points
    
    for i in range(num_control_points):
        angle = torch.tensor(i * angle_step, device=device)
        if i % 2 == 0:
            # Outer points
            radius = outer_radius
        else:
            # Inner points
            radius = inner_radius
        
        control_points[i, 0] = center[0] + radius * torch.cos(angle)
        control_points[i, 1] = center[1] + radius * torch.sin(angle)
    control_points = torch.cat([control_points, control_points[:degree]])
    return control_points

def create_polygon_bspline_control_points(num_vertices, degree=1, device=None):
    """
    Create control points for a regular polygon B-spline curve.
    
    Args:
        num_vertices: int - number of polygon vertices
        degree: int - degree of the B-spline
        device: torch device
    """
    if device is None:
        device = torch.device('cpu')
    
    angles = torch.linspace(0, 2 * torch.pi, num_vertices + 1, device=device)[:-1]  # Exclude duplicate
    x = torch.cos(angles)
    y = torch.sin(angles)
    vertices = torch.stack([x, y], dim=1)
    control_points = torch.cat([vertices, vertices[:degree]])
    return control_points

def create_L_shape_bspline_control_points(degree=1, device=None):
    """
    Create control points for an L-shaped B-spline curve.
    
    Args:
        degree: int - degree of the B-spline
        device: torch device
    """
    if device is None:
        device = torch.device('cpu')
    
    control_points = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.5, 0.5],
        [1.0, 0.5],
        [1.0, 0.0]
    ], device=device)
    control_points = torch.cat([control_points, control_points[:degree]])
    return control_points

def plot_bspline_distance_field(control_points, degree=3, N=400, extent=(-2, 2, -2, 2), 
                               contour=False, levels=20, device=None, chunk_size=200000):
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
    pts_np = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)
    pts = torch.from_numpy(pts_np).to(device)
    
    # Compute distances in chunks
    distances = torch.empty(pts.shape[0], device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            distances[i:j] = bspline_signed_distance_vectorized(
                pts[i:j], control_points, degree=degree, device=device
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

def generate_points_on_curve(control_points, degree=3, num_points=100, device=None, return_t=False):
    torch_device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    control_points = control_points.to(torch_device)
    knots = create_knot_vector(control_points.shape[0], degree, closed=True).to(torch_device)
    min_t = knots[degree]
    max_t = knots[-degree-1]
    t_values = torch.rand(num_points, device=torch_device) * (max_t - min_t) + min_t
    curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device=torch_device)
    if return_t:
        return curve_points, t_values
    else:
        return curve_points

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
    knots = create_knot_vector(control_points.shape[0], degree, closed=True).to(device)
    
    # Sample parameter values for normal vectors
    t_values = torch.linspace(knots[degree], knots[-degree-1], num_vectors, device=device)
    
    # Evaluate curve points
    curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device)
    
    # Compute normal vectors
    normals = bspline_normalvectors(t_values, control_points, knots, degree, device)
    
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

def plot_model_error_map(model, ctrl_pts, degree=1, N=200, extent=(-1.1, 1.1, -1.1, 1.1), use_log=False, device=None):
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
    pts_np = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)
    pts = torch.from_numpy(pts_np).to(device)
    
    # Compute model predictions in chunks
    model_values = torch.empty(pts.shape[0], device=device, dtype=torch.float32)
    chunk_size = 200000
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            model_values[i:j] = model(pts[i:j]).squeeze()
    
    # Compute true signed distances in chunks
    true_distances = torch.empty(pts.shape[0], device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = min(i + chunk_size, pts.shape[0])
            true_distances[i:j] = bspline_signed_distance_vectorized(
                pts[i:j], ctrl_pts, degree=degree, device=device
            )
    plt.figure(figsize=(8, 8))
    # Compute absolute error
    errors = torch.abs(model_values - true_distances)
    Z = errors.cpu().numpy().reshape(N, N)
    if use_log:
        plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    else:
        plt.contourf(X, Y, Z, levels=50, locator=ticker.LogLocator(), cmap='plasma')
    
    
    plt.colorbar(label='Absolute Error')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Error Map')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
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
        control_points = torch.from_numpy(control_points.astype(np.float32))
    control_points = control_points.to(device).float()

    # Create a uniform knot vector (uses existing helper)
    knots = create_knot_vector(control_points.shape[0], degree, closed=closed).to(device)

    # Parameter samples
    t = torch.linspace(knots[degree], knots[-degree-1], num_samples, device=device)

    # Evaluate curve
    curve_points = evaluate_bspline_curve_vectorized(t, control_points, knots, degree, device=device)

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

    return curve_points
if __name__ == "__main__":
    # Run tests
    circle_cp = create_circle_bspline_control_points(center=(0, 0), radius=1.0, num_points=14, degree=2)
    star_cp = create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, 
                                                 num_star_points=5, degree=1)
    #print(circle_cp)
    #plot_bspline_curve(circle_cp, degree=2, closed=True)
    # Uncomment to visualize (requires matplotlib)
    # print("\nGenerating visualizations...")
    #plot_bspline_distance_field(circle_cp, degree=2, N=100, extent=(-2, 2, -2, 2))
    plot_normal_vectors_on_bspline(star_cp, degree=1, num_vectors=100, vector_length=0.2, device=None)
    plot_bspline_distance_field(star_cp, degree=1, N=70, extent=(-2, 2, -2, 2))
    
    # Also run the original star distance plot for comparison
    #plot_star_distance_on_unit_square(N=200, R=1.0, r=0.5, n=5, contour=False, levels=50)