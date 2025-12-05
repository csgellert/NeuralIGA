import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import Geomertry
import math
import geometry_bspline as bsp_geom
import SDF
from matplotlib.animation import FuncAnimation
class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()

        layers = []
        self.loss_history = []
        self.optimizer = None
        self.name = "ReLU"
        self.lr_scheduler = None
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
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
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
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
class Siren_old(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.loss_history = []
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
        self.optimizer = None
        self.name = "SIREN_old"
        self.lr_scheduler = None
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": []
        }

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
class SIRELU(nn.Module):
    def __init__(self, architecture, first_omega_0=60):
        super().__init__()
        self.architecture = architecture
        in_features = architecture[0]
        out_features = architecture[-1]
        hidden_layers = len(architecture)-2

        self.loss_history = []
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
        self.optimizer = None
        self.name = "SIRELU"
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
            self.net.append(nn.Linear(architecture[i+1], architecture[i+2]))
            if i < len(architecture) - 3:
                self.net.append(nn.ReLU())
        self.net.append(nn.Linear(architecture[-2],architecture[-2] ))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(architecture[-2], out_features))
        

        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output
class Siren_SC(nn.Module):
    def __init__(self, architecture, outermost_linear=False,
                 first_omega_0=60, hidden_omega_0=60):
        #siren with skip connection
        super().__init__()
        self.architecture = architecture
        in_features = architecture[0]
        out_features = architecture[-1]
        hidden_layers = len(architecture)-2

        self.loss_history = []
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
        self.optimizer = None
        self.name = "SIREN_SC"
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
        #forward with skip connection
        x_orig = coords
        for i, layer in enumerate(self.net):
            x_new = layer(x)
            if i % 2 == 1 and i < len(self.net) - 2:  # after each SineLayer except the last
                x = x_orig + x_new  # skip connection
            else:
                x = x_new
        return x

class SoftSine(nn.Module):
    def __init__(self, architecture, outermost_linear=False,
                first_omega_0=60, hidden_omega_0=60):
        super().__init__()
        self.architecture = architecture
        in_features = architecture[0]
        out_features = architecture[-1]
        hidden_layers = len(architecture)-2

        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
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
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
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
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
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
        distances = SDF.distance_from_star_contour_vectorized(coordinates, device=device)
        
        return coordinates, distances.view(-1, 1)
    if fun_num == 4: # L-shape
        coordinates = 2 * torch.rand(num_samples, 2, device=device) - 1
        distances = SDF.distance_from_L_shape_vectorized(coordinates, device=device)
        return coordinates, distances.view(-1, 1)
    else:
        raise NotImplementedError
   

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
        import Geomertry
        model = Geomertry.AnaliticalDistanceCircle()
        return model
    elif model_type == "analitical_model2":
        import Geomertry
        model = Geomertry.AnaliticalDistanceLshape()
        return model
    elif model_type == "double_circle_test":
        model = Siren(in_features=2, out_features=1, hidden_features=256,hidden_layers=2, outermost_linear=True, first_omega_0=60, hidden_omega_0=60)
        model.load_state_dict(torch.load('./models/double_circle_test.pth',weights_only=True,map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        raise NotImplementedError("Model not implemented")
def generate_bspline_data(num_samples, case=1, device=None, data_gen_params={}):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        y = bsp_geom.bspline_signed_distance_vectorized(x, star_cp, device=device)
        return x, y.view(-1, 1)
    if case == 2: #pentagon shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        pentagon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, pentagon_cp, degree=1, device=device)
        return x, y.view(-1, 1)
    elif case == 3: #rounded_star
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        rounded_star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=2)
        y = bsp_geom.bspline_signed_distance_vectorized(x, rounded_star_cp, degree=2, device=device)
        return x, y.view(-1, 1)
    elif case == 4: #L-shape
        margain = 0.1
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        L_shape_cp = bsp_geom.create_L_shape_bspline_control_points(degree=1, device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, L_shape_cp, degree=1, device=device)
        return x, y.view(-1, 1)
    elif case == 5: #n-gon
        margain = 0.1
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        x = torch.rand(num_samples, 2, device=device) * (2+margain*2) - 1-margain  # Range [-1, 1]
        n_gon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(x, n_gon_cp, degree=1, device=device)
        return x, y.view(-1, 1)
    else:
        raise NotImplementedError
def generate_bspline_boundary_points(num_boundary_points, case=1, device=None, data_gen_params={}):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=star_cp, num_points=num_boundary_points, degree=1, device=device)
        return boundary_points
    if case == 2: #pentagon shape
        pentagon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=pentagon_cp, num_points=num_boundary_points, degree=1, device=device)
        return boundary_points
    elif case == 3: #rounded_star
        rounded_star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=2)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=rounded_star_cp, num_points=num_boundary_points, degree=2, device=device)
        return boundary_points
    elif case == 4: #L-shape
        L_shape_cp = bsp_geom.create_L_shape_bspline_control_points(degree=1, device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=L_shape_cp, num_points=num_boundary_points, degree=1, device=device)
        return boundary_points
    elif case == 5: #n-gon
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        n_gon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        boundary_points = bsp_geom.generate_points_on_curve(control_points=n_gon_cp, num_points=num_boundary_points, degree=1, device=device)
        return boundary_points
    else:
        raise NotImplementedError
def evaluate_bspline_data_gen(grid, case=1, device=None, data_gen_params={}):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if case == 1: #star shape
        star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, star_cp, degree=1, device=device)
        return y.view(-1, 1)
    if case == 2: #pentagon shape
        pentagon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=5,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, pentagon_cp, degree=1, device=device)
        return y.view(-1, 1)
    if case == 3: #rounded_star
        rounded_star_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=2)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, rounded_star_cp, degree=2, device=device)
        return y.view(-1, 1)
    elif case == 4: #L-shape
        L_shape_cp = bsp_geom.create_L_shape_bspline_control_points(degree=1, device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, L_shape_cp, degree=1, device=device)
        return y.view(-1, 1)
    elif case == 5: #n-gon
        if 'num_vertices' in data_gen_params:
            num_vertices = data_gen_params['num_vertices']
        else:
            num_vertices = 3
        n_gon_cp = bsp_geom.create_polygon_bspline_control_points(num_vertices=num_vertices,degree=1,device=device)
        y = bsp_geom.bspline_signed_distance_vectorized(grid, n_gon_cp, degree=1, device=device)
        return y.view(-1, 1)
    else:
        raise NotImplementedError
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

def train_models_with_extras(model_list, num_epochs = 100, batch_size=10000, fun_num=1, *, device=None,
                              crt = nn.L1Loss(),use_scheduler = False, create_error_history = False,eikon_coeff=0.0,boundry_coeff=0.0,
                              xi_coeff=0.0,boundary_norm_coeff=0.0, evaluation_coeff=1, data_gen_mode='standard',data_gen_params={},
                              create_error_distribution_hystory = False, create_weight_distribution_history = False, hytory_after_epochs = 100, error_distribution_resolution=50,
                              create_SDF_history = False):
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
        if data_gen_mode == 'standard':
            pts, target = generate_data(batch_size, fun_num=fun_num, device=device) 
        elif data_gen_mode == 'bspline':
            pts, target = generate_bspline_data(batch_size, case=fun_num, device=device, data_gen_params=data_gen_params)
        else:
            raise NotImplementedError("Data generation mode not implemented")
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
                #generating boundary points
                bndr_pts = generate_bspline_boundary_points(num_boundary_points=batch_size, case=fun_num, device=device, data_gen_params=data_gen_params)
                bndr_pred = model(bndr_pts)
                boundary_term = (bndr_pred ** 2).mean()
                loss += boundry_coeff * boundary_term
            if xi_coeff > 0.0:
                xi_term = torch.exp(-100 * torch.abs(pred)).mean()
                loss += xi_coeff * xi_term
            if boundary_norm_coeff > 0.0:
                raise NotImplementedError("Boundary norm term not implemented in this snippet.")

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
                        true_values = evaluate_bspline_data_gen(grid_points, case=fun_num, device=device, data_gen_params=data_gen_params)
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
                    elif isinstance(layer, SineLayer):
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
        elif isinstance(layer, SineLayer):
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