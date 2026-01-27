import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import Geomertry
import math
import geometry_bspline as bsp_geom
import geometry_definitions as geom_defs
import network_defs as net_defs
import SDF
from matplotlib.animation import FuncAnimation
def generate_data(num_samples, fun_num=0, device=None, data_gen_params={}):
    """
    Generate training data for various functions.
    
    Args:
        num_samples: number of data points to generate
        fun_num: function type (0=circle, 1=star, 4=L-shape, 5=line)
        device: torch device for computation (CPU/CUDA)
        data_gen_params: dict with additional parameters (e.g., 'angle', 'offset' for line)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if fun_num == 0: # circle
        # Generate random (x, y) coordinates between -1 and 1
        margain = data_gen_params.get('margain', 0.0)
        x = torch.rand(num_samples, 2, device=device) * (2+2*margain) - 1-margain  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y =  1 - torch.sqrt(torch.pow(x[:, 0], 2) + torch.pow(x[:, 1], 2))
        return x, y.view(-1, 1)
    elif fun_num==1: #star shape
        margain = data_gen_params.get('margain', 0.0)
        coordinates = (2+2*margain) * torch.rand(num_samples, 2, device=device) - 1 -margain # Tensor of shape (500, 2) with values in range [-1, 1]

        # Calculate distances for each point using vectorized function
        distances = SDF.distance_from_star_contour_vectorized(coordinates, device=device)
        
        return coordinates, distances.view(-1, 1)
    elif fun_num == 4: # L-shape
        margain = data_gen_params.get('margain', 0.0)
        coordinates = (2+2*margain) * torch.rand(num_samples, 2, device=device) - 1 - margain
        distances = SDF.distance_from_L_shape_vectorized(coordinates, device=device)
        return coordinates, distances.view(-1, 1)
    elif fun_num == 5: # infinite line
        angle = data_gen_params.get('angle', 0.0)  # angle in radians
        offset = data_gen_params.get('offset', 0.0)  # perpendicular offset from origin
        margain = data_gen_params.get('margain', 0.0)
        coordinates = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1 - margain
        distances = SDF.distance_from_line_vectorized(coordinates, angle=angle, offset=offset, device=device)
        return coordinates, distances.view(-1, 1)
    else:
        raise NotImplementedError


def generate_standard_boundary_points(num_boundary_points, fun_num=0, device=None, data_gen_params={}, 
                                       use_importance_sampling=False, importance_sampling_params={}):
    """
    Generate boundary points for standard (non-bspline) geometries.
    
    Args:
        num_boundary_points: int - number of boundary points to generate
        fun_num: function type (0=circle, 1=star, 4=L-shape, 5=line)
        device: torch device for computation
        data_gen_params: dict with additional parameters
        use_importance_sampling: if True, add Gaussian noise to boundary points and return SDF values
        importance_sampling_params: dict with 'sigma' for noise level (default 0.01)
    
    Returns:
        boundary_points: tensor of shape (num_boundary_points, 2)
        sdf_values: tensor of shape (num_boundary_points, 1) if use_importance_sampling, else None
    """
    sigma = importance_sampling_params.get('sigma', 0.01)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if fun_num == 0:  # circle with radius 1
        # Generate points on unit circle
        theta = torch.rand(num_boundary_points, device=device) * 2 * math.pi
        boundary_points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        
        if use_importance_sampling:
            range_min = data_gen_params.get('range_min', -1.005)
            range_max = data_gen_params.get('range_max', 1.005)
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            boundary_points = torch.clamp(boundary_points, range_min, range_max)
            # SDF for circle: 1 - x^2 - y^2
            sdf_val = 1 - torch.sqrt(boundary_points[:, 0]**2 + boundary_points[:, 1]**2)
            return boundary_points, sdf_val.view(-1, 1)
        return boundary_points
    
    elif fun_num == 1:  # star shape
        R = data_gen_params.get('outer_radius', 1.0)
        r = data_gen_params.get('inner_radius', 0.5)
        n = data_gen_params.get('num_star_points', 5)
        
        # Generate points on star contour
        star_vertices = SDF.calculate_star_vertices_vectorized(R, r, n, device)
        num_vertices = star_vertices.shape[0]
        
        # Randomly select edges and positions on those edges
        edge_indices = torch.randint(0, num_vertices, (num_boundary_points,), device=device)
        t = torch.rand(num_boundary_points, device=device)
        
        start_points = star_vertices[edge_indices]
        end_points = star_vertices[(edge_indices + 1) % num_vertices]
        
        # Interpolate along edges
        boundary_points = start_points + t.unsqueeze(1) * (end_points - start_points)
        
        if use_importance_sampling:
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            boundary_points = torch.clamp(boundary_points, -1.1, 1.1)
            sdf_val = SDF.distance_from_star_contour_vectorized(boundary_points, R=R, r=r, n=n, device=device)
            return boundary_points, sdf_val.view(-1, 1)
        return boundary_points
    
    elif fun_num == 4:  # L-shape
        # L-shape corners
        corners = torch.tensor([(-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)], device=device)
        num_corners = corners.shape[0]
        range_min = data_gen_params.get('range_min', -1.005)
        range_max = data_gen_params.get('range_max', 1.005)
        # Randomly select edges and positions
        edge_indices = torch.randint(0, num_corners, (num_boundary_points,), device=device)
        t = torch.rand(num_boundary_points, device=device)
        
        start_points = corners[edge_indices]
        end_points = corners[(edge_indices + 1) % num_corners]
        
        boundary_points = start_points + t.unsqueeze(1) * (end_points - start_points)
        
        if use_importance_sampling:
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            boundary_points = torch.clamp(boundary_points, range_min, range_max)
            sdf_val = SDF.distance_from_L_shape_vectorized(boundary_points, device=device)
            return boundary_points, sdf_val.view(-1, 1)
        return boundary_points
    
    elif fun_num == 5:  # infinite line
        angle = data_gen_params.get('angle', 0.0)
        offset = data_gen_params.get('offset', 0.0)
        range_min = data_gen_params.get('range_min', -1.5)
        range_max = data_gen_params.get('range_max', 1.5)
        
        boundary_points = SDF.generate_points_on_line(num_boundary_points, angle=angle, offset=offset, 
                                                       range_min=range_min, range_max=range_max, device=device)
        
        if use_importance_sampling:
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            boundary_points = torch.clamp(boundary_points, -1.1, 1.1)
            sdf_val = SDF.distance_from_line_vectorized(boundary_points, angle=angle, offset=offset, device=device)
            return boundary_points, sdf_val.view(-1, 1)
        return boundary_points
    
    else:
        raise NotImplementedError(f"Boundary point generation not implemented for fun_num={fun_num}")
def get_standard_SDF_gradient(points, fun_num=0, device=None, data_gen_params={}):
    """
    Compute the gradient of the SDF for standard geometries.
    
    Args:
        points: tensor of shape (N, 2) - input points
        fun_num: function type (0=circle, 1=star, 4=L-shape, 5=line)
        device: torch device for computation
        data_gen_params: dict with additional parameters
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    points = points.to(device)
    
    if fun_num == 0:  # circle
        points.requires_grad_(True)
        sdf_values = 1 - torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    elif fun_num == 1:  # star shape
        raise NotImplementedError("SDF gradient not implemented for star shape")
        sdf_values = SDF.distance_from_star_contour_vectorized(points, device=device)
    elif fun_num == 4:  # L-shape
        closest_point, dist = SDF.get_closest_cntr_point_L_shape_vectorized(points)
        # where dist is zero the gradient shows inside the shape, so we need to handle that

        gradients = torch.zeros_like(points)
        torch_mask = dist.squeeze() == 0
        horiz_mask_up = torch_mask & (points[:,1] == -1)
        horiz_mask_down = torch_mask & ((points[:,1] == 1) | ((points[:,1] == 0) & (points[:,0] > 0)))
        vert_mask_right = torch_mask & (points[:,0] == -1)
        vert_mask_left = torch_mask & ((points[:,0] == 0) | ((points[:,0] == 1) & (points[:,1] > 0)))
        diff = points - closest_point 
        dist_safe = torch.where(dist.unsqueeze(1) == 0, torch.ones_like(dist.unsqueeze(1)), dist.unsqueeze(1))  # to avoid division by zero
        grads = diff / dist_safe

        gradients[~torch_mask] = grads[~torch_mask]
        gradients[horiz_mask_up, :] = torch.tensor([0.0, 1.0], device=device)
        gradients[horiz_mask_down, :] = torch.tensor([0.0, -1.0], device=device)
        gradients[vert_mask_right, :] = torch.tensor([1.0, 0.0], device=device)
        gradients[vert_mask_left, :] = torch.tensor([-1.0, 0.0], device=device)
        return gradients

    elif fun_num == 5:  # infinite line
        angle = data_gen_params.get('angle', 0.0)
        offset = data_gen_params.get('offset', 0.0)
        sdf_values = SDF.distance_from_line_vectorized(points, angle=angle, offset=offset, device=device)
    else:
        raise NotImplementedError(f"SDF gradient not implemented for fun_num={fun_num}")
    
    gradients = torch.autograd.grad(outputs=sdf_values,
                                    inputs=points,
                                    grad_outputs=torch.ones_like(sdf_values),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    
    return gradients

def generate_bspline_data(num_samples, case=1, device=None, data_gen_params={}, gt_num_curve_samples=1000, use_refinement = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        y = bsp_geom.bspline_signed_distance_vectorized(x, star_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return x, y.view(-1, 1)
    if case == 2: #pentagon shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        pentagon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, pentagon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return x, y.view(-1, 1)
    elif case == 3: #rounded_star
        deg = 2
        num_star_points = 5
        if 'degree' in data_gen_params:
            deg = data_gen_params['degree']
        if 'num_star_points' in data_gen_params:
            num_star_points = data_gen_params['num_star_points']
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        rounded_star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=num_star_points, degree=deg)
        y = bsp_geom.bspline_signed_distance_vectorized(x, rounded_star_cp, degree=deg, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return x, y.view(-1, 1)
    elif case == 4: #L-shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        L_shape_cp = geom_defs.create_L_shape_bspline_control_points(degree=1, device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, L_shape_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return x, y.view(-1, 1)
    elif case == 5: #n-gon
        margain = 0.1
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        n_gon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, n_gon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return x, y.view(-1, 1)
    else:
        raise NotImplementedError
def generate_bspline_boundary_points(num_boundary_points, case=1, device=None, data_gen_params={}, use_importance_sampling = False, importance_sampling_params = {}, gt_num_curve_samples=1000, use_refinement = False):
    sigma = 0.01
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=star_cp, num_points=num_boundary_points, degree=1, device=device)
        if use_importance_sampling:
            #adding gaussian noise to boundary points
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            #clipping points to be within [-1, 1]
            boundary_points = torch.clamp(boundary_points, -1.0, 1.0)
            sdf_val = bsp_geom.bspline_signed_distance_vectorized(boundary_points, star_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
            return boundary_points, sdf_val.view(-1, 1)
    elif case == 2: #pentagon shape
        pentagon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=pentagon_cp, num_points=num_boundary_points, degree=1, device=device)
        if use_importance_sampling:
            #adding gaussian noise to boundary points
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            #clipping points to be within [-1, 1]
            boundary_points = torch.clamp(boundary_points, -1.0, 1.0)
            sdf_val = bsp_geom.bspline_signed_distance_vectorized(boundary_points, pentagon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
            return boundary_points, sdf_val.view(-1, 1)
    elif case == 3: #rounded_star
        deg = 2
        num_star_points = 5
        if 'degree' in data_gen_params:
            deg = data_gen_params['degree']
        if 'num_star_points' in data_gen_params:
            num_star_points = data_gen_params['num_star_points']
        rounded_star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=num_star_points, degree=deg)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=rounded_star_cp, num_points=num_boundary_points, degree=deg, device=device)
        if use_importance_sampling:
            #adding gaussian noise to boundary points
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            #clipping points to be within [-1, 1]
            boundary_points = torch.clamp(boundary_points, -1.0, 1.0)
            sdf_val = bsp_geom.bspline_signed_distance_vectorized(boundary_points, rounded_star_cp, degree=deg, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
            return boundary_points, sdf_val.view(-1, 1)
    elif case == 4: #L-shape
        L_shape_cp = geom_defs.create_L_shape_bspline_control_points(degree=1, device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=L_shape_cp, num_points=num_boundary_points, degree=1, device=device)
        if use_importance_sampling:
            #adding gaussian noise to boundary points
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            #clipping points to be within [-1, 1]
            boundary_points = torch.clamp(boundary_points, -1.0, 1.0)
            sdf_val = bsp_geom.bspline_signed_distance_vectorized(boundary_points, L_shape_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
            return boundary_points, sdf_val.view(-1, 1)
    elif case == 5: #n-gon
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        n_gon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=n_gon_cp, num_points=num_boundary_points, degree=1, device=device)
        if use_importance_sampling:
            #adding gaussian noise to boundary points
            noise = torch.randn_like(boundary_points, device=device) * sigma
            boundary_points += noise
            #clipping points to be within [-1, 1]
            boundary_points = torch.clamp(boundary_points, -1.0, 1.0)
            sdf_val = bsp_geom.bspline_signed_distance_vectorized(boundary_points, n_gon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
            return boundary_points, sdf_val.view(-1, 1)
    else:
        raise NotImplementedError
    return boundary_points
def evaluate_bspline_data_gen(grid, case=1, device=None, data_gen_params={}, gt_num_curve_samples=1000, use_refinement = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, star_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return y.view(-1, 1)
    if case == 2: #pentagon shape
        pentagon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, pentagon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return y.view(-1, 1)
    if case == 3: #rounded_star
        deg = 2
        num_star_points = 5
        if 'degree' in data_gen_params:
            deg = data_gen_params['degree']
        if 'num_star_points' in data_gen_params:
            num_star_points = data_gen_params['num_star_points']
        rounded_star_cp = geom_defs.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=num_star_points, degree=deg)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, rounded_star_cp, degree=deg, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return y.view(-1, 1)
    elif case == 4: #L-shape
        L_shape_cp = geom_defs.create_L_shape_bspline_control_points(degree=1, device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, L_shape_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return y.view(-1, 1)
    elif case == 5: #n-gon
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        n_gon_cp = geom_defs.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, n_gon_cp, degree=1, device=device, num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        return y.view(-1, 1)
    else:
        raise NotImplementedError

def get_gradient_error(model, grds_gt, pts,metric='L1'):
    pts.requires_grad_(True)
    pred = model(pts)
    grads = torch.autograd.grad(outputs=pred, inputs=pts,
                                grad_outputs=torch.ones_like(pred),
                                create_graph=True, retain_graph=True)[0]
    # cosine similarity of grads between gt ang model predictions
    cos = nn.CosineSimilarity(dim=1, eps=1e-16)
    similarity = cos(grads, grds_gt)
    similarity_error = 1-similarity
    lengths = torch.norm(grads, dim=1)
    if metric == 'L1':
        mean_similarity = torch.mean(torch.abs(similarity_error)).item()
        length_error = torch.mean(torch.abs(lengths - 1)).item()
    elif metric == 'L_inf':
        mean_similarity = torch.max(similarity_error).item()
        length_error = torch.max(torch.abs(lengths - 1)).item()
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    return length_error, mean_similarity

def train_models_with_extras(model_list, num_epochs = 100, batch_size=10000, fun_num=1, *, device=None,
                              crt = nn.L1Loss(),
                              use_scheduler = False, 
                              eikon_coeff=0.0,boundry_coeff=0.0, xi_coeff=0.0,boundary_norm_coeff=0.0, evaluation_coeff=1, 
                              data_gen_mode='standard',data_gen_params={},
                              create_error_history = False, create_error_distribution_hystory = False, 
                              create_weight_distribution_history = False, hytory_after_epochs = 100, error_distribution_resolution=50,
                              create_SDF_history = False, gt_num_curve_samples = 1000,
                              use_importance_sampling = False, importance_sampling_params = {}, importance_sampling_coeff = 0.0,
                              use_refinement = False,
                              grad_on_bnd_coeff = 0.0, eikon_near_bnd = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    criterion = crt
    if create_error_history:
        crt_L1 = nn.L1Loss()
        crt_L2 = nn.MSELoss()
        crt_Linf = lambda output, target: torch.max(torch.abs(output - target))
    report_interval = max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        if data_gen_mode == 'standard':
            pts, target = generate_data(batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params) 
        elif data_gen_mode == 'bspline':
            pts, target = generate_bspline_data(batch_size, case=fun_num, device=device, data_gen_params=data_gen_params, gt_num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
        else:
            raise NotImplementedError("Data generation mode not implemented")
        for idx, model in enumerate(model_list):
            model.to(device)
            pred = model(pts)
            loss = evaluation_coeff * criterion(pred, target)

            if isinstance(eikon_coeff, (list, tuple)):
                current_eikon_coeff = eikon_coeff[idx]
            else:
                current_eikon_coeff = eikon_coeff
            if isinstance(boundry_coeff, (list, tuple)):
                current_boundry_coeff = boundry_coeff[idx]
            else:
                current_boundry_coeff = boundry_coeff
            if isinstance(importance_sampling_coeff, (list, tuple)):
                current_importance_sampling_coeff = importance_sampling_coeff[idx]
            else:
                current_importance_sampling_coeff = importance_sampling_coeff
            if isinstance(importance_sampling_params, (list, tuple)):
                current_importance_sampling_params = importance_sampling_params[idx]
            if isinstance(grad_on_bnd_coeff, (list, tuple)):
                current_grad_on_bnd_coeff = grad_on_bnd_coeff[idx]
            else:
                current_importance_sampling_params = importance_sampling_params

            if current_eikon_coeff > 0.0:
                # Eikonal term
                pts.requires_grad_(True)
                pred_eik = model(pts)
                grads = torch.autograd.grad(outputs=pred_eik, inputs=pts,
                                            grad_outputs=torch.ones_like(pred_eik),
                                            create_graph=True, retain_graph=True)[0]
                eikonal_term = criterion(grads.norm(dim=1), torch.ones_like(pred_eik))
                
                if eikon_near_bnd:  
                    if data_gen_mode == 'bspline': raise NotImplementedError("Eikonal near boundary not implemented for bspline data generation.")
                    near_bndr_pts, _ = generate_standard_boundary_points(num_boundary_points=batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params, use_importance_sampling=True, importance_sampling_params={'sigma':0.05})
                    near_bndr_pts.requires_grad_(True)
                    near_bndr_pred_eik = model(near_bndr_pts)
                    near_bndr_grads = torch.autograd.grad(outputs=near_bndr_pred_eik, inputs=near_bndr_pts,
                                                          grad_outputs=torch.ones_like(near_bndr_pred_eik),
                                                          create_graph=True, retain_graph=True)[0]
                    near_bndr_eikonal_term = criterion(near_bndr_grads.norm(dim=1), torch.ones_like(near_bndr_pred_eik))
                eikonal_term = eikonal_term  + (near_bndr_eikonal_term if eikon_near_bnd else 0.0)
                eikonal_term = eikonal_term / (1.0  + float(eikon_near_bnd))
                loss += current_eikon_coeff * eikonal_term
            if current_boundry_coeff > 0.0:
                # Generating boundary points based on data generation mode
                if data_gen_mode == 'bspline':
                    bndr_pts = generate_bspline_boundary_points(num_boundary_points=batch_size, case=fun_num, device=device, data_gen_params=data_gen_params)
                else:
                    bndr_pts = generate_standard_boundary_points(num_boundary_points=batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params)
                if current_grad_on_bnd_coeff > 0.0:
                    bndr_pts.requires_grad_(True)
                bndr_pred = model(bndr_pts)
                if current_grad_on_bnd_coeff > 0.0:
                    bndr_grads = torch.autograd.grad(outputs=bndr_pred, inputs=bndr_pts,
                                                    grad_outputs=torch.ones_like(bndr_pred),
                                                    create_graph=True, retain_graph=True)[0]
                    if data_gen_mode == 'bspline': raise NotImplementedError("Gradient on boundary not implemented for bspline data generation.")
                    target_grads = get_standard_SDF_gradient(bndr_pts, fun_num=fun_num, device=device, data_gen_params=data_gen_params)
                    # error v2
                    #cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(bndr_grads, target_grads)
                    #angle_error = torch.mean(1 - cos_sim)
                    #eikon_errror = torch.mean(torch.abs(bndr_grads.norm(dim=1) - 1))
                    #loss += current_grad_on_bnd_coeff * (angle_error + eikon_errror)
                    boundary_grad_term = criterion(bndr_grads, target_grads)
                    loss += current_grad_on_bnd_coeff * boundary_grad_term
                boundary_term = criterion(bndr_pred, torch.zeros_like(bndr_pred))
                loss += current_boundry_coeff * boundary_term
            if xi_coeff > 0.0:
                xi_term = torch.exp(-100 * torch.abs(pred)).mean()
                loss += xi_coeff * xi_term
            if boundary_norm_coeff > 0.0:
                raise NotImplementedError("Boundary norm term not implemented in this snippet.")
            if use_importance_sampling and current_importance_sampling_coeff > 0.0:
                # Importance sampling based on data generation mode
                if data_gen_mode == 'bspline':
                    bndr_pts, targets = generate_bspline_boundary_points(num_boundary_points=batch_size, case=fun_num, device=device, data_gen_params=data_gen_params, use_importance_sampling=True, importance_sampling_params= current_importance_sampling_params, gt_num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
                else:
                    bndr_pts, targets = generate_standard_boundary_points(num_boundary_points=batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params, use_importance_sampling=True, importance_sampling_params=current_importance_sampling_params)
                bndr_pred = model(bndr_pts)
                loss += current_importance_sampling_coeff * criterion(bndr_pred, targets)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.loss_history.append(loss.item())  
            if use_scheduler and model.lr_scheduler is not None:
                model.lr_scheduler.step()
            if create_error_history:
                with torch.no_grad():
                    L1_error = crt_L1(pred, target).item()
                    L2_error = torch.sqrt(crt_L2(pred, target)).item()
                    Linf_error = crt_Linf(pred, target).item()
                    model.error_history["L1"].append(L1_error)
                    model.error_history["L2"].append(L2_error)
                    model.error_history["Linf"].append(Linf_error)  
        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch}], Losses: " + 
                  ", ".join([f"{model.name}: {model.loss_history[-1]}" for model in model_list]))
        if create_error_distribution_hystory and (epoch + 1) % hytory_after_epochs == 0:
            resolution = error_distribution_resolution
            for model in model_list:
                X, Y = torch.meshgrid(torch.linspace(-1, 1, resolution, device=device),
                                      torch.linspace(-1, 1, resolution, device=device))
                grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)
                with torch.no_grad():
                    predictions = model(grid_points)
                    if data_gen_mode == 'standard':
                        raise NotImplementedError("Error distribution history for standard data generation not implemented in this snippet.")
                    elif data_gen_mode == 'bspline':
                        true_values = evaluate_bspline_data_gen(grid_points, case=fun_num, device=device, data_gen_params=data_gen_params, gt_num_curve_samples=gt_num_curve_samples, use_refinement=use_refinement)
                        errors = torch.abs(predictions - true_values).cpu().numpy()
                        error_distribution = errors.reshape(resolution, resolution)
                        model.error_distribution_history.append(error_distribution)
                
        if create_weight_distribution_history and (epoch + 1) % hytory_after_epochs == 0:
            for model in model_list:
                weight_distributions = []
                for layer in model.net:
                    if isinstance(layer, nn.Linear):
                        weights = layer.weight.data.cpu().numpy().flatten()
                        weight_distributions.append(weights)
                    elif isinstance(layer, net_defs.SineLayer):
                        weights = layer.linear.weight.data.cpu().numpy().flatten()
                        weight_distributions.append(weights)
                model.weight_distribution_history.append(weight_distributions)
        if create_SDF_history and (epoch + 1) % hytory_after_epochs == 0:
            resolution = error_distribution_resolution
            for model in model_list:
                X, Y = torch.meshgrid(torch.linspace(-1, 1, resolution, device=device),
                                      torch.linspace(-1, 1, resolution, device=device))
                grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)
                with torch.no_grad():
                    predictions = model(grid_points)
                    sdf_values = predictions.cpu().numpy().reshape(resolution, resolution)
                    model.SDF_history.append(sdf_values)

def train_hotspot(model, fun_num, num_epochs=1000, batch_size=1500, device=None, crt=nn.L1Loss(),
                 bnd_coeff=1.0,bnd_grd_coeff=1.0, hotspot_coeff=1.0,eikon_coeff = 1.0, off_srf_coeff=1.0, data_gen_params={}, hotspot_params={}, use_lr_sched = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    criterion = crt
    report_interval = max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        # Boundary data
        bnd = generate_standard_boundary_points(batch_size,fun_num, device=device)
        bnd.requires_grad_(True)
        model.to(device)
        pred = model(bnd)
        bnd_err = criterion(pred, torch.zeros_like(pred))
        loss = bnd_coeff * bnd_err
        
        bnd_grads = torch.autograd.grad(outputs=pred, inputs=bnd,
                                        grad_outputs=torch.ones_like(pred),
                                        create_graph=True, retain_graph=True)[0]
        target_grads = get_standard_SDF_gradient(bnd, fun_num=fun_num, device=device)
        boundary_grad_term = criterion(bnd_grads, target_grads)
        loss += bnd_grd_coeff * boundary_grad_term
        
        # Hotspot data
        if hotspot_coeff > 0.0:
            #random points in the domain
            margain = data_gen_params.get('margain', 0.05)
            pts = torch.rand(batch_size, 2, device=device) * (2.0+2*margain) - 1.0 - margain # Range [-1, 1]
            pts.requires_grad_(True)
            pred_hotspot = model(pts)
            grad_hotspot = torch.autograd.grad(outputs=pred_hotspot, inputs=pts,
                                              grad_outputs=torch.ones_like(pred_hotspot),
                                              create_graph=True, retain_graph=True)[0]
            grad_length = grad_hotspot.norm(dim=1)
            lambda_hotspot = hotspot_params.get('lambda', 0.1)
            L_heat = torch.mean(0.5*torch.exp(-2*lambda_hotspot*torch.abs(pred_hotspot)) *(grad_length**2 +1)) 
            loss += hotspot_coeff * L_heat
            eikon_loss = criterion(grad_length, torch.ones_like(pred_hotspot))
            loss += eikon_coeff * eikon_loss
        off_srf_punish = torch.exp(-10*torch.abs(pred_hotspot)).mean()
        loss += off_srf_coeff * off_srf_punish

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        model.loss_history.append(loss.item())  
        if use_lr_sched and model.lr_scheduler is not None:
            model.lr_scheduler.step()
        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch}], Loss: {loss.item()}\t(Boundary: {bnd_err.item()}, Boundary Grad: {boundary_grad_term.item()}, Hotspot: {L_heat.item()}, eikonal: {eikon_loss.item()}, Off Surface Punish: {off_srf_punish.item()})")

def R_train(model, fun_num,num_epochs=1000, batch_size=10000, device=None, crt=nn.L1Loss(),
            data_gen_params={}, use_lr_sched = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    criterion = crt
    report_interval = max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        bnd = generate_standard_boundary_points(batch_size, case=fun_num, device=device, data_gen_params=data_gen_params)
        bnd.requires_grad_(True)
        bnd_pred = model(bnd)
        bnd_loss = criterion(bnd_pred, torch.zeros_like(bnd_pred))
        loss += bnd_loss
        bnd_normals = bnd_normals(bnd, fun_num=fun_num, device=device, data_gen_params=data_gen_params)

        #stepssizes
        step_size = torch.rand(batch_size, 1, device=device)
        pts_new = bnd + step_size * bnd_normals
        middle_pred = model(pts_new)
        
        #Rvachev error


        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        model.loss_history.append(loss.item())  
        if use_lr_sched and model.lr_scheduler is not None:
            model.lr_scheduler.step()
        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch}], Loss: {loss.item()}")

def plot_model_weight_per_layer_hyst(model):
    # Plot weight histograms for each layer in the model
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.cpu().numpy().flatten()
            plt.figure(figsize=(8, 4))
            plt.hist(weights, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution for Layer {i} ({layer.__class__.__name__})')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
        elif isinstance(layer, net_defs.SineLayer):
            weights = layer.linear.weight.data.cpu().numpy().flatten()
            plt.figure(figsize=(8, 4))
            plt.hist(weights, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution for SineLayer {i}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

def create_animation_error_distribution(model, interval=200, save_path=None, use_log_scale=False, skip_initial_frames=0, adaptive_scaling=False):
    fig, ax = plt.subplots()
    resolution = model.error_distribution_history[0].shape[0]
    if skip_initial_frames > 0:
        error = model.error_distribution_history[skip_initial_frames:]
    else:
        error = model.error_distribution_history
    if use_log_scale:
        im = ax.imshow(np.log(error[0]), extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
    else:
        im = ax.imshow(error[0], extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
    ax.set_title('Log(Error Distribution Over Training)' if use_log_scale else 'Error Distribution Over Training')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax)


    def update(frame):
        arr = error[frame]
        if use_log_scale:
            arr = np.log(np.maximum(arr, 1e-12))  # prevents log(0)
        if adaptive_scaling:
            im.set_clim(vmin=arr.min(), vmax=arr.max())
        im.set_array(arr)
        ax.set_title(f'Error Distribution at Epoch {frame * 100}')
        return [im]

    ani = FuncAnimation(fig, update, frames=len(error), interval=interval, blit=True)
    if save_path:
        ani.save(save_path, writer='imagemagick')

    plt.show()
def create_animation_error_contourf(model, interval=200, save_path=None, use_log_scale=False, skip_initial_frames=0, adaptive_scaling=False, plot_cntr = False):
    fig, ax = plt.subplots()
    resolution = model.error_distribution_history[0].shape[0]
    X, Y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    if skip_initial_frames > 0:
        error = model.error_distribution_history[skip_initial_frames:]
    else:
        error = model.error_distribution_history
    if use_log_scale:
        Z = np.log(np.maximum(error[0], 1e-12))  # prevents log(0)
    else:
        Z = error[0]
    cont = ax.contourf(X, Y, Z, levels=50, cmap='hot')
    fig.colorbar(cont, ax=ax)
    ax.set_title('Log(Error Contour Over Training)' if use_log_scale else 'Error Contour Over Training')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def update(frame):
        arr = error[frame]
        if use_log_scale:
            arr = np.log(np.maximum(arr, 1e-12))  # prevents log(0)
        ax.clear()
        cont = ax.contourf(X, Y, arr, levels=50, cmap='hot')
        if adaptive_scaling:
            cont.set_clim(vmin=arr.min(), vmax=arr.max())
        if plot_cntr and len(model.SDF_history)>0:
            #plot zero level set
            cntr = ax.contour(X, Y, model.SDF_history[frame], levels=[0], colors='blue', linewidths=2)
        ax.set_title(f'Error Contour at Epoch {frame * 100}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return [cont]

    ani = FuncAnimation(fig, update, frames=len(error), interval=interval, blit=True)
    if save_path:
        ani.save(save_path, writer='imagemagick')

    plt.show()
def create_animation_SDF_contourf(model, interval=200, save_path=None, skip_initial_frames=0, adaptive_scaling=False,plot_cntr = False):
    fig, ax = plt.subplots()
    resolution = model.SDF_history[0].shape[0]
    X, Y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    if skip_initial_frames > 0:
        sdf_values = model.SDF_history[skip_initial_frames:]
    else:
        sdf_values = model.SDF_history
    Z = sdf_values[0]
    cont = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(cont, ax=ax)
    ax.set_title('SDF Contour Over Training')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def update(frame):
        arr = sdf_values[frame]
        ax.clear()
        cont = ax.contourf(X, Y, arr, levels=50, cmap='viridis')
        if adaptive_scaling:
            cont.set_clim(vmin=arr.min(), vmax=arr.max())
        if plot_cntr:
            #plot zero level set
            cntr = ax.contour(X, Y, arr, levels=[0], colors='red', linewidths=2)
        ax.set_title(f'SDF Contour at Epoch {frame * 100}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return [cont]

    ani = FuncAnimation(fig, update, frames=len(sdf_values), interval=interval, blit=True)
    if save_path:
        ani.save(save_path, writer='imagemagick')

    plt.show()

if __name__ == "__main__":
    print(generate_bspline_boundary_points(10, case=1, use_importance_sampling = True))
