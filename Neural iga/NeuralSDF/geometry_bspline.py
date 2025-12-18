import torch
import numpy as np
import matplotlib.pyplot as plt
import geometry_definitions as geom_defs

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

def bspline_basis_second_derivatives(t, knots, degree, device=None):
    """
    Compute second derivatives of B-spline basis functions.
    Vectorized implementation for multiple parameter values.
    
    Args:
        t: tensor of shape (num_t,) - parameter values to evaluate at
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline
        device: torch device
    
    Returns:
        tensor of shape (num_t, num_basis) - second derivative basis function values
    """
    if device is None:
        device = t.device
    
    t = t.to(device)
    knots = knots.to(device)
    
    num_t = t.shape[0]
    num_knots = knots.shape[0]
    num_basis = num_knots - degree - 1
    
    # Compute basis functions of degree-2
    basis_lower = bspline_basis_functions(t, knots, degree - 2, device)
    
    # Initialize second derivative basis functions
    second_deriv_basis = torch.zeros(num_t, num_basis, device=device)
    
    for i in range(num_basis):
        # Compute coefficients for the second derivative formula
        # Using the recursive formula: N''_i,p = p * (N'_i,p-1 / (u_i+p - u_i) - N'_i+1,p-1 / (u_i+p+1 - u_i+1))
        
        # Left term
        if i < num_basis and knots[i + degree] != knots[i] and knots[i + degree - 1] != knots[i]:
            coeff1 = degree * (degree - 1) / ((knots[i + degree] - knots[i]) * (knots[i + degree - 1] - knots[i]))
            second_deriv_basis[:, i] += coeff1 * basis_lower[:, i]
        
        # Middle term (appears twice with different signs)
        if i + 1 < num_basis:
            if knots[i + degree] != knots[i] and knots[i + degree] != knots[i + 1]:
                coeff2 = -degree * (degree - 1) / ((knots[i + degree] - knots[i]) * (knots[i + degree] - knots[i + 1]))
                second_deriv_basis[:, i] -= coeff2 * basis_lower[:, i + 1]
            
            if knots[i + degree + 1] != knots[i + 1] and knots[i + degree] != knots[i + 1]:
                coeff3 = -degree * (degree - 1) / ((knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree] - knots[i + 1]))
                second_deriv_basis[:, i] -= coeff3 * basis_lower[:, i + 1]
        
        # Right term
        if i + 2 < num_basis + 2 and i + 1 < num_basis and knots[i + degree + 1] != knots[i + 1] and knots[i + degree] != knots[i + 1]:
            coeff4 = degree * (degree - 1) / ((knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree] - knots[i + 1]))
            if i + 2 <= basis_lower.shape[1]:
                second_deriv_basis[:, i] += coeff4 * basis_lower[:, i + 2] if i + 2 < basis_lower.shape[1] else 0
    
    return second_deriv_basis

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

def bspline_curvature(t, control_points, knots, degree, device=None):
    """
    Compute curvature of B-spline curve at parameter values t.
    
    The curvature is calculated using the formula:
        κ = |C'(t) x C''(t)| / |C'(t)|³
    
    where C'(t) is the first derivative and C''(t) is the second derivative of the curve.
    For 2D curves, the cross product magnitude is |x'y'' - y'x''|.
    
    Args:
        t: tensor of shape (num_t,) - parameter values to evaluate at
        control_points: tensor of shape (num_cp, 2) - control points [x, y]
        knots: tensor of shape (num_knots,) - knot vector
        degree: int - degree of B-spline (must be >= 2)
        device: torch device
    
    Returns:
        tensor of shape (num_t,) - curvature values at each parameter value
    """
    if degree < 2:
        raise ValueError("Degree must be at least 2 to compute curvature")
    
    if device is None:
        device = control_points.device
    
    t = t.to(device)
    control_points = control_points.to(device)
    knots = knots.to(device)
    
    # Compute first derivative basis functions
    deriv_basis = bspline_basis_derivatives(t, knots, degree, device)  # (num_t, num_basis)
    
    # Compute second derivative basis functions
    second_deriv_basis = bspline_basis_second_derivatives(t, knots, degree, device)  # (num_t, num_basis)
    
    # Compute first derivatives (tangent vectors)
    first_derivs = torch.matmul(deriv_basis, control_points)  # (num_t, 2)
    x_prime = first_derivs[:, 0]
    y_prime = first_derivs[:, 1]
    
    # Compute second derivatives
    second_derivs = torch.matmul(second_deriv_basis, control_points)  # (num_t, 2)
    x_double_prime = second_derivs[:, 0]
    y_double_prime = second_derivs[:, 1]
    
    # Compute cross product magnitude for 2D: |x'y'' - y'x''|
    cross_product = torch.abs(x_prime * y_double_prime - y_prime * x_double_prime)
    
    # Compute magnitude of first derivative: sqrt(x'² + y'²)
    first_deriv_magnitude = torch.sqrt(x_prime**2 + y_prime**2 + 1e-10)  # Add epsilon to avoid division by zero
    
    # Compute curvature: κ = |C' × C''| / |C'|³
    curvature = cross_product / (first_deriv_magnitude**3 + 1e-10)
    
    return curvature

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

def distance_points_to_curve_samples_batched(query_points, curve_points, batch_size=200000, use_refinment=False):
    """
    Compute minimal euclidean distance from each query point to a set of curve samples, in batches.

    Args:
        query_points: (N,2) tensor
        curve_points: (S,2) tensor
        batch_size: number of query points per batch
        use_refinment: if True, refine distances with Newton iterations (not implemented)
    Returns:
        min_distances: (N,) tensor of minimal distances
    """
    device = query_points.device
    dtype = query_points.dtype

    N = query_points.shape[0]
    min_d = torch.empty(N, device=device, dtype=dtype)
    if use_refinment: min_idxs = torch.empty(N, device=device, dtype=torch.long)
    # Ensure same dtype
    curve_points = curve_points.to(device=device, dtype=dtype)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        qb = query_points[i:j].to(device=device, dtype=dtype)
        # torch.cdist is often well-optimized on GPU and CPU
        dists = torch.cdist(qb, curve_points)  # (B, S)
        if use_refinment:
            min_dists, min_indices = torch.min(dists, dim=1)
            min_d[i:j] = min_dists
            min_idxs[i:j] = min_indices
        else:
            min_d[i:j], _ = torch.min(dists, dim=1)
    if use_refinment:
        return min_d, min_idxs

    return min_d, None

def bspline_signed_distance_vectorized(query_points, control_points, degree=3,
                                     num_curve_samples=1000, num_polygon_samples=500,
                                     device=None, point_batch_size=200000,
                                     precomputed_curve_points=None,
                                     precomputed_polygon_vertices=None,
                                     precomputed_knots=None, use_refinment=False):
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
    min_distances, min_indices = distance_points_to_curve_samples_batched(query_points, curve_points, batch_size=point_batch_size, use_refinment=use_refinment)

    if use_refinment:
        
        ref_req_treshold = 1e-2  # Threshold to decide if refinement is needed
        refinment_num_curve_samples = 100  # Samples for refinement
        to_refine_mask = min_distances > ref_req_treshold
        if torch.any(to_refine_mask):
            refine_query_points = query_points[to_refine_mask]
            refine_min_indices = min_indices[to_refine_mask]
            # exclude first and last index to avoid boundary issues
            refine_min_indices = torch.clamp(refine_min_indices, 1, curve_points.shape[0]-2)
            
            # Create refined t_values for each point individually
            refined_distances_list = []
            for idx in refine_min_indices:
                t_start = t_values[idx - 1].item()
                t_end = t_values[idx + 1].item()
                t_refined = torch.linspace(t_start, t_end, refinment_num_curve_samples, device=device, dtype=dtype)
                curve_refined = evaluate_bspline_curve_vectorized(t_refined, control_points, knots, degree, device)
                refined_distances_list.append(torch.min(torch.cdist(refine_query_points[len(refined_distances_list)].unsqueeze(0), curve_refined)))
            
            refined_min_distances = torch.stack(refined_distances_list)
            min_distances[to_refine_mask] = refined_min_distances
    # Batched winding number inside/outside test
    inside_mask = point_in_polygon_winding_number_batched(query_points, polygon_vertices, batch_size=point_batch_size)

    # Apply sign: positive if inside, negative if outside
    signed_distances = torch.where(inside_mask, min_distances, -min_distances)
    
    # Return scalar if input was single point
    if single_point:
        return signed_distances[0]
    
    return signed_distances


def find_closest_point_on_bspline_curve(query_point, control_points, degree=3,
                                       num_curve_samples=1000, device=None):
    """
    Find the parameter t on the B-spline curve closest to the query point.
    """
    if device is None:
        device = query_point.device
    query_point = query_point.to(device)
    control_points = control_points.to(device)
    # Create knot vector for closed curve
    knots = create_knot_vector(control_points.shape[0], degree, closed=True).to(device)
    # Sample curve points
    t_values = torch.linspace(knots[degree], knots[-degree-1], num_curve_samples, device=device)
    curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device)
    # Compute distances
    dists = torch.cdist(query_point.unsqueeze(0), curve_points)  # (1, S)
    min_dists, min_indices = torch.min(dists, dim=1)
    return t_values[min_indices]

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

    return curve_points
if __name__ == "__main__":
    # Run tests
    circle_cp = geom_defs.create_circle_bspline_control_points(center=(0, 0), radius=1.0, num_points=14, degree=2)
    star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, 
                                                 num_star_points=5, degree=1)
    #test curvature with plotting
    degree = 2
    rounded_star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5,num_star_points=5, degree=2)
    knots = create_knot_vector(rounded_star_cp.shape[0], degree=2, closed=True)
    t_test = torch.linspace(knots[degree], knots[-degree-1], 1000)
    curvature_values = bspline_curvature(t_test, rounded_star_cp, knots, degree=2)

    
    plt.figure()
    plt.plot(t_test.cpu().numpy(), curvature_values.cpu().numpy())
    plt.title("Curvature of Rounded Star B-spline Curve (Degree 2)")
    plt.xlabel("Parameter t")
    plt.ylabel("Curvature κ")
    plt.grid()
    plt.show()



    #print(circle_cp)
    #plot_bspline_curve(circle_cp, degree=2, closed=True)
    # Uncomment to visualize (requires matplotlib)
    # print("\nGenerating visualizations...")
    #plot_bspline_distance_field(circle_cp, degree=2, N=100, extent=(-2, 2, -2, 2))
    #plot_normal_vectors_on_bspline(star_cp, degree=1, num_vectors=100, vector_length=0.2, device=None)
    #plot_bspline_distance_field(star_cp, degree=1, N=70, extent=(-2, 2, -2, 2))
    
    # Also run the original star distance plot for comparison
    #plot_star_distance_on_unit_square(N=200, R=1.0, r=0.5, n=5, contour=False, levels=50)