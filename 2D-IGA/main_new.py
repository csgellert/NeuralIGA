import numpy as np
import FEM_new as FEM
from mesh import generateRectangularMesh, getDefaultValues, getDirichletPoints
from math import sqrt,sin,cos,pi
import Geomertry

def loadfunction(x,y):
    return - 2*pi**2*sin(pi*x)*sin(pi*y)
def solutionfunction(x,y):
    return sin(pi*x)*sin(pi*y)

#defining geometry:
p = q = 2
knotvector_u = [0, 0, 0, 0.5, 1, 1, 1]
knotvector_w = [0, 0, 0, 1, 1, 1]
weights= [1,1,1,1,1,1,1,1,1,1,1,1]
ctrlpts = [[[-1, 1],
                    [-1,-1],
                    [-1,-1],
                    [1,-1]],
                    [[-0.65, 1],
                    [-0.7, 0],
                    [0,-0.7],
                    [1,-0.65]],
                    [[0, 1],
                    [0, 0],
                    [0, 0],
                    [1, 0]]]
                   
assert p==q
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1
srf = Geomertry.GenerateSurface(knotvector_u,knotvector_w,p,q,ctrlpts)
Geomertry.plotGeneratedSurface(knotvector_u,knotvector_w,p,q,ctrlpts)
#ke_dim = (p+1)*(q+1)# dimension of the elemntmatrix
K = np.zeros(((NControl_u+p+1)*(NControl_w+q+1),(NControl_u+p+1)*(NControl_w+q+1)))
F = np.zeros((NControl_u+p+1)*(NControl_w+q+1))
for elemx in range(p,p+NControl_u+1):
    for elemy in range(q,q+NControl_w+1):
        Ke,Fe = FEM.elemantBspline(p,q,knotvector_u,knotvector_w,ctrlpts,elemx,elemy,NControl_u,NControl_w,weights,ctrlpts,loadfunction)
        K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,NControl_u,NControl_w)
        print(elemx)
#print(K)
#print(F)
dirichlet = getDirichletPoints(int(sqrt(len(F))))
print(dirichlet)
result = FEM.solve(K,F,dirichlet)
#NURBS.plotNURBSbasisFunction(NControl_u,NControl_w,2,1,weigths,knotvector_u,knotvector_w,p,q,NURBS.R2)
#FEM.visualizeResults_new(ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
FEM.plotAlayticHeatmap(solutionfunction)
FEM.visualizeResultsBspline(result,p,q,knotvector_u,knotvector_w,solutionfunction)
#NURBS.plot_surface(Surfacepoints,ctrlpts)

