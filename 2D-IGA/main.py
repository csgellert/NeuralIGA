import NURBS
import numpy as np
import FEM
from mesh import generateRectangularMesh, getDefaultValues, getDirichletPoints
from math import sqrt,sin,cos,pi

def loadfunction(x,y):
    #return 2-(x**2 + y**2)
    return - pi**2 /2 *cos(pi*x/2)*cos(pi*y/2)
def solutionfunction(x,y):
    return cos(pi*x/2)*cos(pi*y/2)
Nurbs_basis = False
#defining geometry:
default = getDefaultValues(div=4,order=2)
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = generateRectangularMesh(*default)
assert p==q and xDivision == yDivision
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
#print(Surfacepoints)
#ke_dim = (p+1)*(q+1)# dimension of the elemntmatrix
K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
for elemx in range(p,p+xDivision+1):
    for elemy in range(q,q+xDivision+1):
        #Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,ctrlpts)
        if Nurbs_basis:
            Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,ctrlpts)
        else:
            Ke,Fe = FEM.elemantBspline(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None,loadfunction)
        K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        print(elemx)
        #K[elemx-1:ke_dim+elemx-1,elemy-1:ke_dim+elemy-1] += Ke
        #F[elemx-1:elemx+ke_dim-1] += Fe
        pass
#print(K)
#print(F)
dirichlet = getDirichletPoints(int(sqrt(len(F))))
print(dirichlet)
result = FEM.solve(K,F,dirichlet)
#NURBS.plotNURBSbasisFunction(NControl_u,NControl_w,2,1,weigths,knotvector_u,knotvector_w,p,q,NURBS.R2)
#FEM.visualizeResults_new(ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
FEM.plotAlayticHeatmap(solutionfunction)
if Nurbs_basis:
    FEM.visualizeResults(Surfacepoints, ctrlpts, result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
else:
    FEM.visualizeResultsBspline(result,p,q,knotvector_u,knotvector_w,solutionfunction)
#NURBS.plot_surface(Surfacepoints,ctrlpts)

