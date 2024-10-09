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
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            #layers.append(nn.ReLU())
            layers.append(nn.Softplus(100))

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
    
def generate_data(num_samples,fun_num = 0):
    if fun_num == 0: # circle
        # Generate random (x, y) coordinates between -1 and 1
        x = torch.rand(num_samples, 2) * 2 - 1  # Range [-1, 1]
        # Compute the target function: 1 - x^2 - y^2
        y = 1 - x[:, 0] ** 2 - x[:, 1] ** 2
        return x, y.view(-1, 1)
    else:
        raise NotImplementedError
    
    

def plotDisctancefunction(eval_fun, N=500,contour = False):
    x_values = np.linspace(-1.1, 1.1, N)
    y_values = np.linspace(-1.1, 1.1, N)
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