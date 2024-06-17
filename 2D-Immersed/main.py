import NURBS
import numpy as np
import FEM
import mesh
from tqdm import tqdm
import time

Nurbs_basis = False
r=1
#defining geometry:
default = mesh.getDefaultValues(div=10,order=1,delta=0.05)
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
assert p==q and xDivision == yDivision
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1
mesh.plotMesh(xDivision,yDivision,delta=0.05)
mesh.plotAlayticHeatmap(FEM.solution_function)
Surfacepoints = []
if Nurbs_basis:
    for xx in x:
        srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
        Surfacepoints.append(srf)


K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
print("Initialisation finished")
for elemx in tqdm(range(p,p+xDivision+1)):
    for elemy in range(q,q+xDivision+1):
        if Nurbs_basis:
            Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,ctrlpts)
        else:
            Ke,Fe = FEM.elementChoose(False,r,p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None)
        K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)

start = time.time()
print("Solving equation")
result = FEM.solveWeak(K,F)
print(f"Calculation time: {time.time()-start} ms")

if Nurbs_basis:
    FEM.visualizeResults(Surfacepoints, ctrlpts, result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
else:
    FEM.visualizeResultsBspline(result,p,q,knotvector_u,knotvector_w,None)#NURBS.getCircularDomain()


