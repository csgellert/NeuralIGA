import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
import Geomertry
import math
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, num_hidden_layers=1, output_size=1):
        super(NeuralNetwork, self).__init__()

        # Create a list to store all layers
        layers = []

        # First layer (input to the first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layers (number of hidden layers is flexible)
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            #layers.append(nn.Softplus(100))

        # Output layer (from the last hidden layer to the output)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine the layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, coords):
        return self.network(coords)
    def train_network(self,optim,total_steps,steps_til_summary=20):
        for step in range(total_steps):
            model_input, ground_truth = generate_data(10000,0)
            model_output = self.forward(model_input)
            loss = ((model_output - ground_truth)**2).mean()


            if not step % steps_til_summary:
                print("Step %d, Total loss %0.8f" % (step, loss))

            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Step %d, Total loss %0.6f" % (step, loss))
    def save(self,use_date = False):
        if use_date:
            date=datetime.datetime.now()
            print(date.strftime('%b'),date.strftime('%d') ,"-",  date.strftime('%X'))
            torch.save(self.state_dict(), f"relu_model{date.strftime('%b')}-{date.strftime('%d')}-{date.strftime('%X')}.pth")
        else:
            torch.save(self.state_dict(), "relu_model_last.pth")
        print("Model saved successfully")
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

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        #self.net.append(nn.Linear(hidden_features,hidden_features))
        #self.net.append(nn.Softplus(100))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
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
def plotDisctancefunction(eval_fun, N=500,contour = False):
    x_values = np.linspace(0, 1.05, N)
    y_values = np.linspace(0, 1.05, N)
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

def benchmark_star_distance_functions(num_points=10000, device=None):
    """
    Benchmark the performance difference between original and vectorized star distance functions.
    """
    import time
    import numpy as np
    import torch
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmarking with {num_points} points on device: {device}")
    
    # Generate test coordinates
    coordinates = 2 * torch.rand(num_points, 2, device=device) - 1
    
    # Test original function (moved to CPU for fairness since original uses Python loops)
    coordinates_cpu = coordinates.cpu()
    print("\nTesting original function...")
    start_time = time.time()
    distances_original = torch.empty(num_points, 1)
    for i in range(num_points):
        x, y = coordinates_cpu[i]
        distances_original[i] = distance_from_star_contour([x.item(), y.item()])
    original_time = time.time() - start_time
    
    # Test vectorized function
    print("Testing vectorized function...")
    start_time = time.time()
    distances_vectorized = distance_from_star_contour_vectorized(coordinates, device=device)
    vectorized_time = time.time() - start_time
    
    # Compare results (move to CPU for comparison)
    distances_vectorized_cpu = distances_vectorized.cpu()
    max_diff = torch.max(torch.abs(distances_original.squeeze() - distances_vectorized_cpu)).item()
    
    print(f"\nResults:")
    print(f"Original function time: {original_time:.4f} seconds")
    print(f"Vectorized function time: {vectorized_time:.4f} seconds")
    print(f"Speedup: {original_time/vectorized_time:.2f}x")
    print(f"Maximum difference in results: {max_diff:.8f}")
    print(f"Results match: {max_diff < 1e-6}")
    
    return original_time, vectorized_time, max_diff
def plot_star_distance_on_unit_square(N=400, R=1.0, r=0.5, n=5, device=None, contour=False, levels=20, cmap='viridis', chunk_size=200000):
    """
    Plot signed distance from the star contour on the unit square [0,1]x[0,1] using the
    distance_from_star_contour_vectorized function.

    Args:
        N: resolution per axis (total points = N*N)
        R, r, n: star parameters forwarded to the distance function
        device: torch device (defaults to cuda if available)
        contour: if True use contour (lines) otherwise filled contourf
        levels: contour levels
        cmap: matplotlib colormap
        chunk_size: process this many points per batch to limit memory use
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create grid in numpy for plotting axes
    x_vals = np.linspace(-1.0, 1.0, N)
    y_vals = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Build points tensor (flattened)
    pts_np = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)  # (N*N, 2)
    pts = torch.from_numpy(pts_np).to(device)

    # Compute distances in chunks to avoid memory spike
    distances = torch.empty(pts.shape[0], device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            j = i + chunk_size
            distances[i:j] = distance_from_star_contour_vectorized(pts[i:j], R=R, r=r, n=n, device=device)

    Z = distances.cpu().numpy().reshape(N, N)

    plt.figure(figsize=(6, 6))
    if contour:
        plt.contour(X, Y, Z, levels=levels, cmap=cmap)
    else:
        plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    
    plt.colorbar(label='signed distance')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Star signed distance (R={R}, r={r}, n={n}) on unit square')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(X, Y, Z, levels=[0], colors='red')
    plt.show()
# =====================================================================================
# B-SPLINE SIGNED DISTANCE FUNCTION
# =====================================================================================

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
    More robust than ray casting for complex polygons.
    
    Args:
        points: tensor of shape (num_points, 2) - query points
        polygon_vertices: tensor of shape (num_vertices, 2) - polygon vertices
    
    Returns:
        tensor of shape (num_points,) - boolean mask (True if inside)
    """
    num_points = points.shape[0]
    num_vertices = polygon_vertices.shape[0]
    
    winding_numbers = torch.zeros(num_points, device=points.device)
    
    for i in range(num_vertices):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % num_vertices]
        
        # Check if edge crosses upward
        upward = (v1[1] <= points[:, 1]) & (points[:, 1] < v2[1])
        
        # Check if edge crosses downward  
        downward = (v2[1] <= points[:, 1]) & (points[:, 1] < v1[1])
        
        # For crossing edges, check if point is left of edge
        for j in range(num_points):
            if upward[j]:
                # Compute cross product to determine side
                cross = (v2[0] - v1[0]) * (points[j, 1] - v1[1]) - (v2[1] - v1[1]) * (points[j, 0] - v1[0])
                if cross > 0:  # Point is left of edge
                    winding_numbers[j] += 1
            elif downward[j]:
                cross = (v2[0] - v1[0]) * (points[j, 1] - v1[1]) - (v2[1] - v1[1]) * (points[j, 0] - v1[0])
                if cross < 0:  # Point is left of edge
                    winding_numbers[j] -= 1
    
    return winding_numbers != 0

def bspline_signed_distance_vectorized(query_points, control_points, degree=3, 
                                     num_curve_samples=1000, num_polygon_samples=500, 
                                     device=None):
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
    
    query_points = query_points.to(device)
    control_points = control_points.to(device)
    
    # Compute minimum distances to curve
    min_distances = distance_points_to_bspline_curve_vectorized(
        query_points, control_points, degree, num_curve_samples, device
    )
    
    # Determine inside/outside using polygon approximation
    knots = create_knot_vector(control_points.shape[0], degree, closed=True)
    knots = knots.to(device)
    
    # Sample polygon vertices for inside/outside test
    t_polygon = torch.linspace(0, 1, num_polygon_samples + 1, device=device)[:-1]  # Exclude duplicate endpoint
    polygon_vertices = evaluate_bspline_curve_vectorized(t_polygon, control_points, knots, degree, device)
    
    # Test if points are inside
    inside_mask = point_in_polygon_winding_number(query_points, polygon_vertices)
    
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

def generate_points_on_curve(control_points, degree=3, num_points=100, device=None):
    torch_device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    control_points = control_points.to(torch_device)
    knots = create_knot_vector(control_points.shape[0], degree, closed=True).to(torch_device)
    min_t = knots[degree]
    max_t = knots[-degree-1]
    t_values = torch.rand(num_points, device=torch_device) * (max_t - min_t) + min_t
    curve_points = evaluate_bspline_curve_vectorized(t_values, control_points, knots, degree, device=torch_device)
    return curve_points

# =====================================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================================

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
    plot_bspline_distance_field(star_cp, degree=1, N=70, extent=(-2, 2, -2, 2))
    
    # Also run the original star distance plot for comparison
    #plot_star_distance_on_unit_square(N=200, R=1.0, r=0.5, n=5, contour=False, levels=50)