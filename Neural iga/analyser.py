import NURBS
import numpy as np
import FEM
import mesh
from tqdm import tqdm
import time
import torch
from NeuralImplicit import NeuralNetwork
import matplotlib.pyplot as plt
from math import pi,sin,cos

# def loadfunction(x,y):
#     #return 2-(x**2 + y**2)
#     return - pi**2 /2 *cos(pi*x/2)*cos(pi*y/2)
# def solutionfunction(x,y):
#     return cos(pi*x/2)*cos(pi*y/2)
relu_model = NeuralNetwork(2,256,2,1)
relu_model.load_state_dict(torch.load('relu_model_last.pth',weights_only=True))
relu_model.eval()
r=1

test_values = [3,5,7,9]
esize = [1/(nd+1) for nd in test_values]
orders = [1,2,3]
fig,ax = plt.subplots()
for order in orders:
    accuracy = []
    for division in test_values:
        default = mesh.getDefaultValues(div=division,order=order,delta=0.05)
        x0, y0,x1,y1,xDivision,yDivision,p,q = default
        knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
        assert p==q and xDivision == yDivision
        x = np.linspace(x0,x1,10)
        y = np.linspace(y0,y1,10)
        NControl_u = len(knotvector_u)-p-1
        NControl_w = len(knotvector_w)-q-1
        
        K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
        F = np.zeros((xDivision+p+1)*(yDivision+q+1))
        for elemx in tqdm(range(p,p+xDivision+1)):
            for elemy in range(q,q+xDivision+1):
                Ke,Fe = FEM.elementChoose(relu_model,False,r,p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None)
                K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        #print(dirichlet)
        result = FEM.solveWeak(K,F)
        accuracy.append(FEM.calculateErrorBspline(relu_model,result,p,q,knotvector_u, knotvector_w))
    #ax.semilogy(test_values,accuracy)
    ax.loglog(esize,accuracy)
ax.set_title("Convergence of MSE based on number of elements")
#ax.set_xlabel("Number of divisions")
ax.set_xlabel("log(Element size)")
ax.set_ylabel("log(Mean Square Error)")
ax.legend(["p=1","p=2","p=3"])

plt.show()