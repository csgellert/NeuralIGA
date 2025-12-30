import torch
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

if __name__ == "__main__":
    # Example usage
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    R = 1
    circle_cp = create_circle_bspline_control_points(center=(0, 0), radius=R, num_points=8, degree=3, device=device)
    #circle_cp = create_polygon_bspline_control_points(num_vertices=6, degree=3, device=device)
    import geometry_bspline as bsp_geom
    import matplotlib.pyplot as plt
    import geometry_visualisation as geom_vis
    import numpy as np
    #print("Circle Control Points:\n", circle_cp)

    geom_vis.plot_bspline_curve(circle_cp, degree=3, num_samples=100,closed=True,show_control_points=True)
    geom_vis.plot_normal_vectors_on_bspline(circle_cp,degree=3)
    plt.axis('equal')
    
    #testing distance comparism with analitycal
    test_points = torch.rand((1000, 2), device=device) * 2.0 - 1.0  # Random points in [-1, 1]x[-1, 1]
    sdf_vals_basic = bsp_geom.bspline_signed_distance_vectorized(test_points, circle_cp, degree=3, device=device, num_curve_samples=1000)
    sdf_vals_high_res = bsp_geom.bspline_signed_distance_vectorized(test_points, circle_cp, degree=3, device=device, num_curve_samples=6000)
    sdf_vals_refined = bsp_geom.bspline_signed_distance_vectorized(test_points, circle_cp, degree=3, device=device, num_curve_samples=1000, use_refinement=False)
    # Analitycal SDF for circle
    test_points_np = test_points.cpu().numpy()
    an_sdf_vals = R - np.sqrt(test_points_np[:, 0]**2 + test_points_np[:, 1]**2)
    # Compare
    error_basic = np.abs(sdf_vals_basic.cpu().numpy().flatten() - an_sdf_vals)
    print("Max SDF Error:", np.max(error_basic))
    print("Mean SDF Error:", np.mean(error_basic))
    error_high_res = np.abs(sdf_vals_high_res.cpu().numpy().flatten() - an_sdf_vals)
    print("Max SDF Error (high res):", np.max(error_high_res))
    print("Mean SDF Error (high res):", np.mean(error_high_res))
    error_refined = np.abs(sdf_vals_refined.cpu().numpy().flatten() - an_sdf_vals)
    print("Max SDF Error (refined):", np.max(error_refined))
    print("Mean SDF Error (refined):", np.mean(error_refined))

    #vsiualize sdf
    geom_vis.plot_bspline_distance_field(circle_cp, degree=3, N=100, device=device, use_refinement=True)