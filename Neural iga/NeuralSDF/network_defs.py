import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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

def save_model(model, path):
    torch.save(model.state_dict(), path)
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class KANLinear(nn.Module):
    """A minimal Kolmogorov-Arnold Network (KAN) linear layer.

    Learns a univariate spline per (out,in) edge (piecewise-linear on a uniform
    grid) and sums contributions over input dims. Optionally adds a standard
    linear term applied to a base activation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 16,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        spline_order: str = "linear",
        bias: bool = True,
        use_base: bool = True,
        base_activation: str = "silu",
        spline_init_scale: float = 1e-2,
    ):
        super().__init__()

        spline_order = str(spline_order).lower().strip()

        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        if spline_order not in ("linear", "cubic"):
            raise ValueError(f"spline_order must be 'linear' or 'cubic', got {spline_order!r}")
        if spline_order == "cubic" and grid_size < 4:
            raise ValueError(
                f"grid_size must be >= 4 for cubic interpolation, got {grid_size}"
            )
        x_min, x_max = float(grid_range[0]), float(grid_range[1])
        if not (x_max > x_min):
            raise ValueError(f"grid_range must satisfy max>min, got {grid_range}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = spline_order
        self.register_buffer("grid_min", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("grid_max", torch.tensor(x_max, dtype=torch.float32))

        self.use_base = bool(use_base)
        self.base_activation = str(base_activation).lower()

        if self.use_base:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.base_bias = nn.Parameter(torch.empty(out_features)) if bias else None
        else:
            self.base_weight = None
            self.base_bias = None

        self.spline_coeff = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.spline_bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.reset_parameters(spline_init_scale=spline_init_scale)

    def reset_parameters(self, spline_init_scale: float = 1e-2):
        with torch.no_grad():
            if self.use_base:
                nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
                if self.base_bias is not None:
                    fan_in = self.base_weight.shape[1]
                    bound = 1 / np.sqrt(fan_in)
                    self.base_bias.uniform_(-bound, bound)

            self.spline_coeff.normal_(mean=0.0, std=float(spline_init_scale))
            if self.spline_bias is not None:
                self.spline_bias.zero_()

    def _apply_base_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_activation in ("identity", "none", "linear"):
            return x
        if self.base_activation in ("silu", "swish"):
            return F.silu(x)
        if self.base_activation == "tanh":
            return torch.tanh(x)
        if self.base_activation == "relu":
            return F.relu(x)
        raise ValueError(
            f"Unknown base_activation={self.base_activation!r}. "
            "Use 'silu', 'tanh', 'relu', or 'identity'."
        )

    def _spline_interpolate(self, x: torch.Tensor) -> torch.Tensor:
        x_min = self.grid_min.to(dtype=x.dtype, device=x.device)
        x_max = self.grid_max.to(dtype=x.dtype, device=x.device)

        x_clipped = torch.clamp(x, x_min, x_max)
        u = (x_clipped - x_min) / (x_max - x_min) * (self.grid_size - 1)

        coeff = self.spline_coeff.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        gather_shape = (x.shape[0], self.out_features, self.in_features, 1)

        def _gather_at(idx: torch.Tensor) -> torch.Tensor:
            idx_g = idx.unsqueeze(1).unsqueeze(-1).expand(gather_shape)
            return torch.gather(coeff, dim=3, index=idx_g).squeeze(-1)

        if self.spline_order == "linear":
            idx0 = torch.floor(u).to(dtype=torch.long)
            idx0 = torch.clamp(idx0, 0, self.grid_size - 2)
            idx1 = idx0 + 1
            t = (u - idx0.to(dtype=u.dtype)).clamp(0.0, 1.0)

            v0 = _gather_at(idx0)
            v1 = _gather_at(idx1)
            t_ = t.unsqueeze(1)
            v = v0 * (1.0 - t_) + v1 * t_
        else:
            i = torch.floor(u).to(dtype=torch.long)
            i = torch.clamp(i, 1, self.grid_size - 3)
            t = (u - i.to(dtype=u.dtype)).clamp(0.0, 1.0)

            p0 = _gather_at(i - 1)
            p1 = _gather_at(i)
            p2 = _gather_at(i + 1)
            p3 = _gather_at(i + 2)

            t_ = t.unsqueeze(1)
            t2 = t_ * t_
            t3 = t2 * t_

            v = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t_
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )

        y = v.sum(dim=2)
        if self.spline_bias is not None:
            y = y + self.spline_bias
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._spline_interpolate(x)
        if self.use_base:
            xb = self._apply_base_activation(x)
            y = y + F.linear(xb, self.base_weight, self.base_bias)
        return y


class KAN(nn.Module):
    """A simple multi-layer KAN built from `KANLinear` layers."""

    def __init__(
        self,
        architecture,
        grid_size: int = 16,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        spline_order: str = "linear",
        use_base: bool = True,
        base_activation: str = "silu",
        bias: bool = True,
        spline_init_scale: float = 1e-2,
    ):
        super().__init__()
        self.architecture = architecture

        self.loss_history = []
        self.error_distribution_history = []
        self.weight_distribution_history = []
        self.SDF_history = []
        self.optimizer = None
        self.name = "KAN"
        self.lr_scheduler = None
        self.error_history = {
            "L1": [],
            "L2": [],
            "Linf": [],
        }

        layers = []
        for i in range(len(architecture) - 1):
            layers.append(
                KANLinear(
                    architecture[i],
                    architecture[i + 1],
                    grid_size=grid_size,
                    grid_range=grid_range,
                    spline_order=spline_order,
                    bias=bias,
                    use_base=use_base,
                    base_activation=base_activation,
                    spline_init_scale=spline_init_scale,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)