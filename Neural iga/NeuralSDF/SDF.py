import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def distance_from_line_vectorized(coords, angle=0.0, offset=0.0, device=None):
    """
    Compute signed distance from points to an infinite line passing through origin.
    
    The line is defined by its angle from the positive x-axis.
    Points on the positive side (counter-clockwise from the line) have positive distance.
    
    Args:
        coords: tensor of shape (num_points, 2) or (2,) - [x, y] coordinates
        angle: float - angle of the line in radians (0 = horizontal, pi/2 = vertical)
        offset: float - perpendicular offset of the line from origin
        device: torch device for computation
    
    Returns:
        tensor of shape (num_points,) or scalar - signed distances
    """
    # Handle single point input
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
        single_point = True
    else:
        single_point = False
    
    if device is None:
        device = coords.device
    
    coords = coords.to(device)
    
    # Normal vector to the line (perpendicular, pointing to positive side)
    # Line direction is (cos(angle), sin(angle))
    # Normal is (-sin(angle), cos(angle))
    normal_x = -math.sin(angle)
    normal_y = math.cos(angle)
    
    # Signed distance = dot product with normal - offset
    # d = x * (-sin(angle)) + y * cos(angle) - offset
    signed_distances = coords[:, 0] * normal_x + coords[:, 1] * normal_y - offset
    
    if single_point:
        return signed_distances[0]
    
    return signed_distances


def generate_points_on_line(num_points, angle=0.0, offset=0.0, range_min=-1.0, range_max=1.0, device=None):
    """
    Generate random points on an infinite line within a specified range.
    
    Args:
        num_points: int - number of points to generate
        angle: float - angle of the line in radians
        offset: float - perpendicular offset of the line from origin
        range_min: float - minimum parameter value along the line
        range_max: float - maximum parameter value along the line
        device: torch device
    
    Returns:
        tensor of shape (num_points, 2) - points on the line
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random parameter values along the line
    t = torch.rand(num_points, device=device) * (range_max - range_min) + range_min
    
    # Line direction
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    
    # Point on line closest to origin (at perpendicular distance = offset)
    base_x = -offset * math.sin(angle)
    base_y = offset * math.cos(angle)
    
    # Generate points
    points = torch.zeros(num_points, 2, device=device)
    points[:, 0] = base_x + t * dir_x
    points[:, 1] = base_y + t * dir_y
    
    return points


def l_shape_distance(crd):
    """
    Calculates the distance of a point (x, y) from an L-shaped domain.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.

    Returns:
        The distance of the point from the L-shaped domain.
    """

    x = crd[0]
    y = crd[1]
    corners = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0),(1.0,0.5),(0.5,0.5),(0.5,1.0)]
    dists = [distance_point_to_line(x,y,corners[i][0], corners[i][1], corners[i+1][0],corners[i+1][1]) for i in range(len(corners)-1)]
    dists.append(distance_point_to_line(x,y,corners[-1][0], corners[-1][1], corners[0][0],corners[0][1]))
    dist = min(dists)

    sgn1 = 1 if x>=0 and x<1 and y>=0 and y<0.5 else -1
    sgn2 = 1 if x>=0 and x<0.5 and y>=0.5 and y<=1 else -1
    sgn = max(sgn1,sgn2)


    return dist*sgn
def calculate_star_vertices(R, r, n=5):
    vertices = []
    angle_between_points = 2 * math.pi / (2 * n)
    for i in range(2 * n):
        if i % 2 == 0:
            radius = R  # Outer point
        else:
            radius = r  # Inner point
        angle = i * angle_between_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y))
    return vertices

def calculate_star_vertices_vectorized(R, r, n=5, device=None):
    """Vectorized version using PyTorch tensors"""
    if device is None:
        device = torch.device('cpu')
    
    num_vertices = 2 * n
    angle_between_points = 2 * torch.pi / num_vertices
    
    # Create indices for all vertices
    indices = torch.arange(num_vertices, device=device)
    
    # Calculate radii: outer points (even indices) get R, inner points (odd indices) get r
    radii = torch.where(indices % 2 == 0, torch.tensor(R, device=device), torch.tensor(r, device=device))
    
    # Calculate angles
    angles = indices * angle_between_points
    
    # Calculate vertices
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    
    # Stack into vertices tensor of shape (num_vertices, 2)
    vertices = torch.stack([x, y], dim=1)
    
    return vertices

def is_point_inside_star(x, y, star_vertices):
    num_vertices = len(star_vertices)
    inside = False
    for i in range(num_vertices):
        x1, y1 = star_vertices[i]
        x2, y2 = star_vertices[(i + 1) % num_vertices]

        # Check if point (x, y) is inside the star by ray-casting
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def is_points_inside_star_vectorized(points, star_vertices):
    """
    Vectorized ray-casting algorithm to check if points are inside the star.
    
    Args:
        points: tensor of shape (num_points, 2) - [x, y] coordinates
        star_vertices: tensor of shape (num_vertices, 2) - star vertices
    
    Returns:
        tensor of shape (num_points,) - boolean mask indicating which points are inside
    """
    num_points = points.shape[0]
    num_vertices = star_vertices.shape[0]
    
    # Extract x, y coordinates
    x = points[:, 0]  # (num_points,)
    y = points[:, 1]  # (num_points,)
    
    # Initialize inside mask
    inside = torch.zeros(num_points, dtype=torch.bool, device=points.device)
    
    # For each edge of the star
    for i in range(num_vertices):
        x1 = star_vertices[i, 0]
        y1 = star_vertices[i, 1]
        x2 = star_vertices[(i + 1) % num_vertices, 0]
        y2 = star_vertices[(i + 1) % num_vertices, 1]
        
        # Ray-casting algorithm: check if horizontal ray from point crosses this edge
        # Condition 1: edge crosses the horizontal line at y-level of the point
        crosses_y = (y1 > y) != (y2 > y)
        
        # Condition 2: intersection point is to the right of the point
        # Calculate x-coordinate of intersection
        denom = y2 - y1
        # Avoid division by zero (should not happen due to crosses_y condition)
        denom = torch.where(torch.abs(denom) < 1e-10, torch.ones_like(denom) * 1e-10, denom)
        intersection_x = (x2 - x1) * (y - y1) / denom + x1
        crosses_right = x < intersection_x
        
        # If both conditions are met, toggle the inside status
        edge_crossing = crosses_y & crosses_right
        inside = inside ^ edge_crossing  # XOR operation (toggle)
    
    return inside

def distance_from_star_contour(coord, R=1, r=0.5, n=5):
    x=coord[0]
    y=coord[1]
    # Get star vertices
    vertices = calculate_star_vertices(R, r, n)

    # Calculate the minimum distance from point to the star contour
    min_distance = float('inf')
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        dist = distance_point_to_line(x, y, x1, y1, x2, y2)
        min_distance = min(min_distance, dist)

    # Determine if point is inside or outside the star
    inside = is_point_inside_star(x, y, vertices)

    # Return positive distance if inside, negative if outside
    return min_distance if inside else -min_distance

def distance_from_star_contour_vectorized(coords, R=1, r=0.5, n=5, device=None):
    """
    Vectorized version that processes multiple coordinates simultaneously.
    
    Args:
        coords: tensor of shape (num_points, 2) or (2,) for single point - [x, y] coordinates
        R: outer radius of star
        r: inner radius of star  
        n: number of star points
        device: torch device for computation
    
    Returns:
        tensor of shape (num_points,) or scalar - signed distances (positive inside, negative outside)
    """
    # Handle single point input
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
        single_point = True
    else:
        single_point = False
    
    if device is None:
        device = coords.device
    
    # Ensure coords is on the correct device
    coords = coords.to(device)
    
    # Get star vertices
    star_vertices = calculate_star_vertices_vectorized(R, r, n, device)
    
    # Create line segments from vertices
    num_vertices = star_vertices.shape[0]
    line_starts = star_vertices
    line_ends = torch.roll(star_vertices, shifts=-1, dims=0)  # Next vertex for each vertex
    
    # Calculate distances from all points to all line segments
    distances = distance_points_to_line_segments_vectorized(coords, line_starts, line_ends)
    
    # Find minimum distance to any edge for each point
    min_distances = torch.min(distances, dim=1)[0]  # (num_points,)
    
    # Determine if points are inside or outside the star
    inside_mask = is_points_inside_star_vectorized(coords, star_vertices)
    
    # Apply sign: positive if inside, negative if outside
    signed_distances = torch.where(inside_mask, min_distances, -min_distances)
    
    # Return scalar if input was single point
    if single_point:
        return signed_distances[0]
    
    return signed_distances
def distance_from_L_shape_vectorized(coords, device=None):
    """
    Vectorized calculation of distances from multiple points to an L-shaped domain.
    
    Args:
        coords: tensor of shape (num_points, 2) - [x, y] coordinates
    
    Returns:
        tensor of shape (num_points,) - signed distances (positive inside, negative outside)
    """
    num_points = coords.shape[0]
    
    # Define L-shape corners
    corners = torch.tensor([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0)], device=coords.device)
    
    # Create line segments from corners
    line_starts = corners
    line_ends = torch.roll(corners, shifts=-1, dims=0)  # Next corner for each corner
    
    # Calculate distances from all points to all line segments
    distances = distance_points_to_line_segments_vectorized(coords, line_starts, line_ends)
    
    # Find minimum distance to any edge for each point
    min_distances = torch.min(distances, dim=1)[0]  # (num_points,)
    
    # Determine sign based on position relative to L-shape
    x = coords[:, 0]
    y = coords[:, 1]
    
    sgn1 = torch.where((x >= 0) & (x < 1) & (y >= 0) & (y < 0.5), torch.tensor(1.0, device=coords.device), torch.tensor(-1.0, device=coords.device))
    sgn2 = torch.where((x >= 0) & (x < 0.5) & (y >= 0.5) & (y <= 1), torch.tensor(1.0, device=coords.device), torch.tensor(-1.0, device=coords.device))
    sgn = torch.max(sgn1, sgn2)
    
    signed_distances = min_distances * sgn
    
    return signed_distances

def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate the perpendicular distance from point (px, py) to the line segment (x1, y1) -> (x2, y2)."""
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_sq == 0:  # The segment is a point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def distance_points_to_line_segments_vectorized(points, line_starts, line_ends):
    """
    Vectorized calculation of distances from multiple points to multiple line segments.
    
    Args:
        points: tensor of shape (num_points, 2) - [x, y] coordinates
        line_starts: tensor of shape (num_segments, 2) - starting points of line segments
        line_ends: tensor of shape (num_segments, 2) - ending points of line segments
    
    Returns:
        tensor of shape (num_points, num_segments) - distances from each point to each segment
    """
    num_points = points.shape[0]
    num_segments = line_starts.shape[0]
    
    # Expand dimensions for broadcasting
    # points: (num_points, 1, 2)
    # line_starts: (1, num_segments, 2)
    # line_ends: (1, num_segments, 2)
    points_expanded = points.unsqueeze(1)  # (num_points, 1, 2)
    line_starts_expanded = line_starts.unsqueeze(0)  # (1, num_segments, 2)
    line_ends_expanded = line_ends.unsqueeze(0)  # (1, num_segments, 2)
    
    # Calculate line vectors
    line_vecs = line_ends_expanded - line_starts_expanded  # (1, num_segments, 2)
    
    # Calculate vectors from line starts to points
    point_vecs = points_expanded - line_starts_expanded  # (num_points, num_segments, 2)
    
    # Calculate line lengths squared
    line_length_sq = torch.sum(line_vecs ** 2, dim=2)  # (1, num_segments)
    
    # Handle zero-length segments
    line_length_sq = torch.clamp(line_length_sq, min=1e-10)
    
    # Calculate projection parameter t
    dot_product = torch.sum(point_vecs * line_vecs, dim=2)  # (num_points, num_segments)
    t = torch.clamp(dot_product / line_length_sq, min=0.0, max=1.0)  # (num_points, num_segments)
    
    # Calculate projection points
    t_expanded = t.unsqueeze(2)  # (num_points, num_segments, 1)
    projections = line_starts_expanded + t_expanded * line_vecs  # (num_points, num_segments, 2)
    
    # Calculate distances
    distances = torch.norm(points_expanded - projections, dim=2)  # (num_points, num_segments)
    
    return distances
def plotDisctancefunction(eval_fun, N=500, extent=(-1.1, 1.1, -1.1, 1.1), contour = False):
    x_values = np.linspace(extent[0], extent[1], N)
    y_values = np.linspace(extent[2], extent[3], N)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros((N,N))
    # Evaluate the function at each point in the grid
    for idxx, xx in enumerate(x_values):
        for idxy,yy in enumerate(y_values):
            #print(yy)
            try:
                crd = torch.tensor([xx, yy], dtype=torch.float64)
                ans = eval_fun(crd)
                Z[idxx, idxy] = ans[0].item()
            except:
                Z[idxx, idxy] = eval_fun([xx,yy])

    #Z = distanceFromContur(X, Y)

    # Create a contour plot
    if contour:
         plt.contour(X, Y, Z,levels=20)
    else:
        plt.contourf(X, Y, Z,levels=20)
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scalar-Valued Function f(x, y)')
    plt.grid(True)
    plt.show()


def visualize_standard_testcase(fun_num=0, N=200, extent=(-1.1, 1.1, -1.1, 1.1), 
                                 contour=False, plot_zero_level=True, data_gen_params={},
                                 device=None, ax=None, title=None, cmap='viridis'):
    """
    Visualize standard test cases from NeuralImplicit.
    
    Args:
        fun_num: int - test case number
            0: circle (SDF = 1 - sqrt(x^2 + y^2))
            1: star shape (5-pointed star with R=1, r=0.5)
            4: L-shape
            5: infinite line (requires 'angle' and 'offset' in data_gen_params)
        N: int - grid resolution
        extent: tuple - (xmin, xmax, ymin, ymax) for the plot
        contour: bool - if True, use contour lines; if False, use filled contours
        plot_zero_level: bool - if True, highlight the zero level set (boundary)
        data_gen_params: dict - additional parameters for test cases (e.g., angle/offset for line)
        device: torch device
        ax: matplotlib axis - if provided, plot on this axis; otherwise create new figure
        title: str - custom title; if None, auto-generate based on fun_num
        cmap: str - colormap name
    
    Returns:
        fig, ax if ax was None; otherwise just ax
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate grid
    x_values = torch.linspace(extent[0], extent[1], N, device=device)
    y_values = torch.linspace(extent[2], extent[3], N, device=device)
    X, Y = torch.meshgrid(x_values, y_values, indexing='ij')
    coords = torch.stack([X.ravel(), Y.ravel()], dim=1)
    
    # Compute SDF based on fun_num
    if fun_num == 0:  # circle
        Z = 1 - torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        default_title = 'Circle SDF: $1 - \\sqrt{x^2 + y^2}$'
    elif fun_num == 1:  # star shape
        R = data_gen_params.get('outer_radius', 1.0)
        r = data_gen_params.get('inner_radius', 0.5)
        n = data_gen_params.get('num_star_points', 5)
        Z = distance_from_star_contour_vectorized(coords, R=R, r=r, n=n, device=device)
        default_title = f'Star SDF (R={R}, r={r}, n={n})'
    elif fun_num == 4:  # L-shape
        Z = distance_from_L_shape_vectorized(coords, device=device)
        default_title = 'L-shape SDF'
    elif fun_num == 5:  # infinite line
        angle = data_gen_params.get('angle', 0.0)
        offset = data_gen_params.get('offset', 0.0)
        Z = distance_from_line_vectorized(coords, angle=angle, offset=offset, device=device)
        angle_deg = math.degrees(angle)
        default_title = f'Line SDF (angle={angle_deg:.1f}°, offset={offset:.2f})'
    else:
        raise NotImplementedError(f"Visualization not implemented for fun_num={fun_num}")
    
    # Reshape to grid
    Z = Z.reshape(N, N).cpu().numpy()
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Create figure if no axis provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        created_fig = True
    
    # Plot
    if contour:
        cf = ax.contour(X_np, Y_np, Z, levels=20, cmap=cmap)
    else:
        cf = ax.contourf(X_np, Y_np, Z, levels=50, cmap=cmap)
    
    # Add colorbar
    if created_fig:
        plt.colorbar(cf, ax=ax, label='SDF value')
    
    # Plot zero level set (boundary)
    if plot_zero_level:
        ax.contour(X_np, Y_np, Z, levels=[0], colors='red', linewidths=2, linestyles='solid')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title if title else default_title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if created_fig:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return ax


def visualize_all_standard_testcases(N=150, data_gen_params_list=None, device=None):
    """
    Visualize all standard test cases in a single figure.
    
    Args:
        N: int - grid resolution
        data_gen_params_list: list of dicts - parameters for each test case (indexed by fun_num)
        device: torch device
    """
    if data_gen_params_list is None:
        data_gen_params_list = {
            0: {},  # circle
            1: {},  # star
            4: {},  # L-shape
            5: {'angle': math.pi/4, 'offset': 0.0},  # 45° line
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    test_cases = [0, 1, 4, 5]
    titles = ['Circle', 'Star', 'L-shape', 'Line (45°)']
    
    for i, (fun_num, title) in enumerate(zip(test_cases, titles)):
        params = data_gen_params_list.get(fun_num, {})
        visualize_standard_testcase(fun_num=fun_num, N=N, ax=axes[i], 
                                    title=title, data_gen_params=params, device=device)
    
    plt.tight_layout()
    plt.show()
    return fig, axes
