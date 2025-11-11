import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
import Geomertry
import math
import geometry_bspline as bsp_geom
class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()

        layers = []
        self.loss_history = []
        self.optimizer = None
        self.name = "ReLU"
        self.lr_scheduler = None
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }

        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:
                layers.append(nn.ReLU())
        # Combine the layers into a sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
    
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, architecture, outermost_linear=False,
                 first_omega_0=60, hidden_omega_0=60):
        super().__init__()
        self.architecture = architecture
        in_features = architecture[0]
        out_features = architecture[-1]
        hidden_layers = len(architecture)-2

        self.loss_history = []
        self.optimizer = None
        self.name = "SIREN"
        self.lr_scheduler = None
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }

        self.net = []
        self.net.append(SineLayer(in_features, architecture[1],
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers-1):
            self.net.append(SineLayer(architecture[i+1],architecture[i+2] ,
                                      is_first=False, omega_0=hidden_omega_0))
        self.net.append(SineLayer(architecture[-2],architecture[-2] ,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(architecture[-2], out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / architecture[-2]) / hidden_omega_0,
                                              np.sqrt(6 / architecture[-2]) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(architecture[-2], out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output
class SoftSine(nn.Module):
    def __init__(self, architecture, outermost_linear=False,
                first_omega_0=60, hidden_omega_0=60):
        super().__init__()
        self.architecture = architecture
        in_features = architecture[0]
        out_features = architecture[-1]
        hidden_layers = len(architecture)-2

        self.loss_history = []
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }
        self.optimizer = None
        self.name = "SoftSine"
        self.lr_scheduler = None

        self.net = []
        self.net.append(SineLayer(in_features, architecture[1],
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers-1):
            if i % 2 ==0:
                self.net.append(nn.Linear(architecture[i+1],architecture[i+2]))
                self.net.append(nn.Softplus())
            else:
                self.net.append(SineLayer(architecture[i+1],architecture[i+2] ,
                                      is_first=False, omega_0=hidden_omega_0))
        if hidden_layers % 2 == 1:
            self.net.append(nn.Linear(architecture[-2],architecture[-2]))
            self.net.append(nn.Softplus())
        else:
            self.net.append(SineLayer(architecture[-2],architecture[-2] ,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(architecture[-2], out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / architecture[-2]) / hidden_omega_0,
                                              np.sqrt(6 / architecture[-2]) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(architecture[-2], out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

class PosEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)

    def forward(self, x):
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)

class PE_Relu(nn.Module):
    def __init__(self, architecture, num_freqs=10, include_input=True):
        super().__init__()
        self.pos_encoding = PosEncoding(num_freqs, include_input)
        in_features = architecture[0]
        encoded_features = in_features * (2 * num_freqs + int(include_input))
        new_architecture = [encoded_features] + architecture[1:]
        self.net = []
        self.loss_history = []
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }
        self.optimizer = None
        self.name = "PE_ReLU"
        self.lr_scheduler = None
        for i in range(len(new_architecture) - 1):
            self.net.append(nn.Linear(new_architecture[i], new_architecture[i+1]))
            if i < len(new_architecture) - 2:
                self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        encoded_coords = self.pos_encoding(coords)
        output = self.net(encoded_coords)
        return output
class PE_Siren(nn.Module):
    def __init__(self, architecture, num_freqs=10, include_input=True,
                 outermost_linear=False, first_omega_0=60, hidden_omega_0=60):
        super().__init__()
        self.pos_encoding = PosEncoding(num_freqs, include_input)
        in_features = architecture[0]
        encoded_features = in_features * (2 * num_freqs + int(include_input))
        new_architecture = [encoded_features] + architecture[1:]
        self.net = []
        in_features = new_architecture[0]
        out_features = new_architecture[-1]
        hidden_layers = len(new_architecture)-2

        self.loss_history = []
        self.optimizer = None
        self.name = "PE_SIREN"
        self.lr_scheduler = None
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }


        self.net.append(SineLayer(in_features, new_architecture[1],
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers-1):
            self.net.append(SineLayer(new_architecture[i+1],new_architecture[i+2] ,
                                      is_first=False, omega_0=hidden_omega_0))
        self.net.append(SineLayer(new_architecture[-2],new_architecture[-2] ,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(new_architecture[-2], out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / new_architecture[-2]) / hidden_omega_0,
                                              np.sqrt(6 / new_architecture[-2]) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(new_architecture[-2], out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        

    def forward(self, coords):
        encoded_coords = self.pos_encoding(coords)
        output = self.net(encoded_coords)
        return output
def generate_data(num_samples,fun_num = 0, device=None):
    """
    Generate training data for various functions.
    
    Args:
        num_samples: number of data points to generate
        fun_num: function type (0=circle, 1=star, 2=circle_euklidean, 4=L-shape)
        device: torch device for computation (CPU/CUDA)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if fun_num == 0: # circle
        # Generate random (x, y) coordinates between -1 and 1
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain) - 1-margain  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y =  1 - x[:, 0] ** 2 - x[:, 1] ** 2
        return x, y.view(-1, 1)
    elif fun_num==1: #star shape
        coordinates = 2 * torch.rand(num_samples, 2, device=device) - 1  # Tensor of shape (500, 2) with values in range [-1, 1]

        # Calculate distances for each point using vectorized function
        distances = distance_from_star_contour_vectorized(coordinates, device=device)
        
        return coordinates, distances.view(-1, 1)
    elif fun_num ==2: #circle_aukl
        # Generate random (x, y) coordinates between -1 and 1
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain) - 1-margain  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y =  x[:, 0] ** 2 + x[:, 1] ** 2
        dst = torch.sqrt(y)
        y = 1-dst
        return x, y.view(-1, 1)
    if fun_num == 4: # L-shape
        coordinates = 2 * torch.rand(num_samples, 2) - 1  # Tensor of shape (500, 2) with values in range [-1, 1]

        # Calculate distances for each point
        distances = torch.empty(num_samples,1)
        for i in range(num_samples):
            x, y = coordinates[i]
            distances[i] = l_shape_distance([x.item(), y.item()])

        return coordinates, distances
    if fun_num == 5: # L-shape quadratic
        coordinates = 2 * torch.rand(num_samples, 2) - 1  # Tensor of shape (500, 2) with values in range [-1, 1]

        # Calculate distances for each point
        distances = torch.empty(num_samples,1)
        for i in range(num_samples):
            x, y = coordinates[i]
            distances[i] = l_shape_distance([x.item(), y.item()])
        distances = torch.sign(distances)*distances**2
        return coordinates, distances
    else:
        raise NotImplementedError
   
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
                crd = torch.tensor([xx, yy], dtype=torch.float32)
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

def load_models(model_type="siren_model"):
    """options: siren_model, siren_model_kor_jo, siren_model_L-shape, siren_model_L-shape_qvad,siren_model_euk, analitical_model, analitical_model2"""
    print(f"Loading model: {model_type}")

    # Load the model
    #relu_model = NeuralNetwork(2,256,2,1)
    #relu_model.load_state_dict(torch.load('relu_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
    #relu_model.eval()
    if model_type == "siren_model":
        model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
        model.load_state_dict(torch.load('./models/siren_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    elif model_type == "siren_model_kor_jo":
        model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
        model.load_state_dict(torch.load('./models/siren_model_kor_jo.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    elif model_type == "siren_model_euk":
        model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
        model.load_state_dict(torch.load('./models/siren_model_euk_last.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    elif model_type == "siren_model_L-shape":
        model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
        model.load_state_dict(torch.load('./models/siren_model_L-shape.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    elif model_type == "siren_model_L-shape_qvad":
        model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
        model.load_state_dict(torch.load('./models/siren_model_L-shape_qvad.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    elif model_type == "analitical_model":
        model = Geomertry.AnaliticalDistanceCircle()
        return model
    elif model_type == "analitical_model2":
        model = Geomertry.AnaliticalDistanceLshape()
        return model
    elif model_type == "double_circle_test":
        model = Siren(in_features=2, out_features=1, hidden_features=256,hidden_layers=2, outermost_linear=True, first_omega_0=60, hidden_omega_0=60)
        model.load_state_dict(torch.load('./models/double_circle_test.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        raise NotImplementedError("Model not implemented")
def generate_bspline_data(num_samples, case=1, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1:
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain) - 1-margain  # Range [-1, 1]
        star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        y = bsp_geom.bspline_signed_distance_vectorized(x, star_cp, device=device)
        return x, y.view(-1, 1)
def train_models(model_list, num_epochs = 100, batch_size=10000, fun_num=1, device=None, crt = nn.L1Loss(),create_error_history = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = crt
    if create_error_history:
        crt_L1 = nn.L1Loss()
        crt_L2 = nn.MSELoss()
        crt_Linf = lambda output, target: torch.max(torch.abs(output - target))
    report_interval = max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        # Generate training data
        #pts, target = generate_bspline_data(batch_size, case=fun_num, device=device)
        pts, target = generate_data(batch_size, fun_num=fun_num, device=device) 
        for model in model_list:
            model.to(device)
            pred = model(pts)
            loss = criterion(pred, target)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.loss_history.append(loss.item())      
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
                  ", ".join([f"{model.name}: {model.loss_history[-1]:.6f}" for model in model_list]))

def train_models_with_extras(model_list, num_epochs = 100, batch_size=10000, fun_num=1, device=None,
                              crt = nn.L1Loss(),create_error_history = False,eikon_coeff=0.0,boundry_coeff=0.0,
                              xi_coeff=0.0,boundary_norm_coeff=0.0, evaluation_coeff=1):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    criterion = crt
    if create_error_history:
        crt_L1 = nn.L1Loss()
        crt_L2 = nn.MSELoss()
        crt_Linf = lambda output, target: torch.max(torch.abs(output - target))
    report_interval = max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        # Generate training data
        #pts, target = generate_bspline_data(batch_size, case=fun_num, device=device)
        pts, target = generate_data(batch_size, fun_num=fun_num, device=device) 
        for model in model_list:
            model.to(device)
            pred = model(pts)
            loss = evaluation_coeff * criterion(pred, target)
            if eikon_coeff > 0.0:
                # Eikonal term
                pts.requires_grad_(True)
                pred_eik = model(pts)
                grads = torch.autograd.grad(outputs=pred_eik, inputs=pts,
                                            grad_outputs=torch.ones_like(pred_eik),
                                            create_graph=True, retain_graph=True)[0]
                eikonal_term = ((grads.norm(dim=1) - 1) ** 2).mean()
                loss += eikon_coeff * eikonal_term
            if boundry_coeff > 0.0:
                raise NotImplementedError("Boundary term not implemented in this snippet.")
            if xi_coeff > 0.0:
                xi_term = torch.exp(-100 * torch.abs(pred)).mean()
                loss += xi_coeff * xi_term
            if boundary_norm_coeff > 0.0:
                raise NotImplementedError("Boundary norm term not implemented in this snippet.")

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.loss_history.append(loss.item())      
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
                  ", ".join([f"{model.name}: {model.loss_history[-1]:.6f}" for model in model_list]))
