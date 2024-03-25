import NURBS
import numpy as np
import FEM
from mesh import generateRectangularMesh, getDefaultValues
#defining geometry:
p= 1
q = 1
default = getDefaultValues()
knotvector_u, knotvector_w,weigths, ctrlpts = generateRectangularMesh(*default)

x = np.linspace(0,1,10)
y = np.linspace(0,1,7)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-p-1

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
#print(Surfacepoints)

K = np.zeros((4,4))
F = np.zeros(4)
for i in range(1,3):
    for j in range(1,3):
        Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,i,j,NControl_u,NControl_w,weigths)
        K += Ke
        F += Fe
        pass
print(K)
print(np.linalg.inv(K))
print(F)
result = FEM.solve(K,F)
#NURBS.plotNURBSbasisFunction(NControl_u,NControl_w,2,1,weigths,knotvector_u,knotvector_w,p,q,NURBS.R2)
FEM.visualizeResults(Surfacepoints,ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)

NURBS.plot_surface(Surfacepoints,ctrlpts)

