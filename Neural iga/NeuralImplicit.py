import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime

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
            torch.save(self.state_dict(), "siren_model_last.pth")
        print("Model saved successfully")
    def get_model_mse(self,fun_num = 0,n=20000):
        model_input, ground_truth = generate_data(n,fun_num)
        model_output = self.forward(model_input)
        #print(model_input[0])
        loss = ((model_output - ground_truth)**2).mean()
        return loss
def generate_data(num_samples,fun_num = 0):
    if fun_num == 0: # circle
        # Generate random (x, y) coordinates between -1 and 1
        margain = 0.1
        x = torch.rand(num_samples, 2) * (2+margain) - 1-margain  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y =  1 - x[:, 0] ** 2 - x[:, 1] ** 2
        return x, y.view(-1, 1)
    elif fun_num==1: #star shape
        coordinates = 2 * torch.rand(num_samples, 2) - 1  # Tensor of shape (500, 2) with values in range [-1, 1]

        # Calculate distances for each point
        distances = torch.empty(num_samples,1)
        for i in range(num_samples):
            x, y = coordinates[i]
            distances[i] = distance_from_star_contour([x.item(), y.item()])

        return coordinates, distances
    elif fun_num ==2: #circle_aukl
        # Generate random (x, y) coordinates between -1 and 1
        margain = 0.1
        x = torch.rand(num_samples, 2) * (2+margain) - 1-margain  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y =  x[:, 0] ** 2 + x[:, 1] ** 2
        dst = torch.sqrt(y)
        y = 1-dst
        return x, y.view(-1, 1)
    else:
        raise NotImplementedError
   

def plotDisctancefunction(eval_fun, N=500,contour = False):
    x_values = np.linspace(-1, 1, N)
    y_values = np.linspace(-1, 1, N)
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