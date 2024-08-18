import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.manual_seed(42)

# kívül szükséges egy harmadik paraméter, az initnek: a belső (rejtett) tenzor mérete
class MLP(nn.Module):# multi layer perceptron
    def __init__(self, n, m, hidden1, hidden2):
        super().__init__()# constructor of parent class
        #layers
        self.lin0 = nn.Linear(in_features = n, out_features = hidden1)
        self.lin1 = nn.Linear(in_features = hidden1, out_features = hidden2)
        self.lin2 = nn.Linear(in_features = hidden2, out_features = m)
    def forward(self, x):
        x = F.relu(self.lin0(x))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
model = MLP(2,1,64,64) #create instance
optimizer = optim.Adam(model.parameters(),lr= 1e-6)
x = torch.randn(2)

#* Training --------------
def lossFunction(points,pred):
    x = points[0]
    y = points[1]
    r2 = 1 - x**2 - y**2
    predicted = pred
    return (r2-predicted**2)**2


def train(model, optimizer):
    epochs = 120
    training_steps = 500
    model.train()
    runing_loss = 0.0
    losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        runing_loss = 0
        for i in range(training_steps):
            point = 2*torch.rand(2)-1
            output = model(point)
            loss = lossFunction(point,output)

            loss.backward()
            optimizer.step()
            runing_loss += loss.item()
        losses.append(runing_loss)
    return losses
print("start training...")
losses = train(model,optimizer)
plt.plot(losses)
plt.show()
test = torch.tensor([1.0,0.0])
print(model(test))
print(lossFunction(test,model(test)))
