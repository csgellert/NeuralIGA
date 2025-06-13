import NURBS
import numpy as np
import FEM
import mesh
from tqdm import tqdm
import time
import torch
from NeuralImplicit import NeuralNetwork, Siren
import cProfile
from pstats import Stats, SortKey
import Geomertry
# Load the model
#relu_model = NeuralNetwork(2,256,2,1)
#relu_model.load_state_dict(torch.load('relu_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
#relu_model.eval()
siren_model = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
siren_model.load_state_dict(torch.load('siren_model_last.pth',weights_only=True,map_location=torch.device('cpu')))
siren_model.eval()
siren_model_kor_jo = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
siren_model_kor_jo.load_state_dict(torch.load('siren_model_kor_jo.pth',weights_only=True,map_location=torch.device('cpu')))
siren_model_kor_jo.eval()
siren_model_L_shape = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
siren_model_L_shape.load_state_dict(torch.load('siren_model_L-shape.pth',weights_only=True,map_location=torch.device('cpu')))
siren_model_L_shape.eval()
siren_model_L_shape2 = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
siren_model_L_shape2.load_state_dict(torch.load('siren_model_L-shape_qvad.pth',weights_only=True,map_location=torch.device('cpu')))
siren_model_L_shape2.eval()
model = siren_model_L_shape
r=1
#defining geometry:
default = mesh.getDefaultValues(div=30,order=1,delta=0.005,larger_domain=True)
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
assert p==q and xDivision == yDivision
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1
#mesh.plotMesh(xDivision,yDivision,delta=0.005)
#mesh.plotAlayticHeatmap(FEM.solution_function)
Geomertry.init_spl(x,p,None,knotvector_u)

K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
print("Initialisation finished")
with cProfile.Profile() as pr:
    for elemx in tqdm(range(p,p+xDivision+1)):
        for elemy in range(q,q+xDivision+1):
            Ke,Fe = FEM.elementChoose(model,False,r,p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None)
            K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
with open('profiling_stats.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('.prof_stats')
    stats.print_stats()

start = time.time()
print("Solving equation")
result = FEM.solveWeak(K,F)
print(f"Calculation time: {time.time()-start} ms")
#FEM.plotErrorHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=False,N=40)
FEM.plotResultHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=True,N=50)


FEM.visualizeResultsBspline(model,result,p,q,knotvector_u,knotvector_w,None,larger_domain=True)



