import NURBS
import numpy as np
import FEM
import mesh
from tqdm import tqdm
import time
import torch 
from NeuralImplicit import NeuralNetwork, Siren
import matplotlib.pyplot as plt
from math import pi,sin,cos
import Geomertry

# def loadfunction(x,y):
#     #return 2-(x**2 + y**2)
#     return - pi**2 /2 *cos(pi*x/2)*cos(pi*y/2)
# def solutionfunction(x,y):
#     return cos(pi*x/2)*cos(pi*y/2)
relu_model = NeuralNetwork(2,256,2,1)
print(sum(p.numel() for p in relu_model.parameters() if p.requires_grad))
relu_model.load_state_dict(torch.load('relu_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
relu_model.eval()
siren_model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
siren_model.load_state_dict(torch.load('siren_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
siren_model.eval()
model = siren_model
r=1

test_values = [30,50]
esize = [1/(nd+1) for nd in test_values]
orders = [1,2,3]
fig,ax = plt.subplots()
for order in orders:
    accuracy = []
    etypes = []
    for division in test_values:
        etype = {"outer":0,"inner":0,"boundary":0}
        default = mesh.getDefaultValues(div=division,order=order,delta=0.005)
        x0, y0,x1,y1,xDivision,yDivision,p,q = default
        knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
        assert p==q and xDivision == yDivision

        x = np.linspace(x0,x1,10)
        y = np.linspace(y0,y1,10)
        NControl_u = len(knotvector_u)-p-1
        NControl_w = len(knotvector_w)-q-1
        Geomertry.init_spl(x,p,None,knotvector_u)
        
        K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
        F = np.zeros((xDivision+p+1)*(yDivision+q+1))
        for elemx in tqdm(range(p,p+xDivision+1)):
            for elemy in range(q,q+xDivision+1):
                Ke,Fe,etype = FEM.elementChoose(model,False,r,p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None,etype)
                K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        #print(dirichlet)
        result = FEM.solveWeak(K,F)
        accuracy.append(FEM.calculateErrorBspline(model,result,p,q,knotvector_u, knotvector_w))
        etypes.append(etype)
    #ax.semilogy(test_values,accuracy)
    ax.loglog(esize,accuracy)
ax.set_title("Convergence of MSE based on number of elements")
#ax.set_xlabel("Number of divisions")
ax.set_xlabel("log(Element size)")
ax.set_ylabel("log(Mean Square Error)")
ax.legend(["p=1","p=2","p=3"])

plt.show()
inner = np.array([case["inner"] for case in etypes])
outer = np.array([case["outer"] for case in etypes])
boundary = np.array([case["boundary"] for case in etypes])
print(etypes)
fig,ax = plt.subplots()
ax.loglog(esize,(inner+boundary)/(inner+boundary+outer))
plt.show()