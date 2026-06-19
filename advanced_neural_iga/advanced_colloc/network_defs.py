import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ReLU_network(nn.Module):
    def __init__(self, architecture):
        super(ReLU_network, self).__init__()

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
            out.append(torch.sin(freq *torch.pi * x))
            out.append(torch.cos(freq *torch.pi * x))

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

def load_test_model(name, type, params = {}):
    if type == "SIREN":
        architecture = params.get("architecture", [2, 256, 256, 256,  1])
        w_0 = params.get("w_0", 30)
        w_hidden = params.get("w_hidden", 30)
        model = Siren(architecture=architecture, outermost_linear=True, first_omega_0=w_0, hidden_omega_0=w_hidden)
        model.load_state_dict(torch.load(f"trained_models/{name}.pth", map_location=torch.device('cpu'), weights_only=True))
        model = model.double()  # Convert to float64
        model.eval()
        return model
    elif type == "ReLU":
        architecture = params.get("architecture", [2, 256, 256, 256, 256, 1])
        model = ReLU_network(architecture=architecture)
        model.load_state_dict(torch.load(f"trained_models/{name}.pth", map_location=torch.device('cpu'), weights_only=True))
        model = model.double()  # Convert to float64
        model.eval()
        return model
    elif type == "PE_ReLU":
        model = PE_Relu(architecture=[2, 256, 256, 256, 256, 1], num_freqs=10, include_input=True)
    else:
        raise ValueError(f"Unknown model type: {type}")
    return model