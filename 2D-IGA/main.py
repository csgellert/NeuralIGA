import NURBS
import numpy as np
import FEM
from mesh import generateRectangularMesh, getDefaultValues
#defining geometry:
default = getDefaultValues()
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = generateRectangularMesh(*default)
assert p==1 and q==1
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,7)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-p-1

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
#print(Surfacepoints)
ke_dim = (p+1)*(q+1)# dimension of the elemntmatrix
K = np.zeros(((xDivision+1+1)*(yDivision+1+1),(xDivision+1+1)*(yDivision+1+1)))
F = np.zeros((xDivision+1+1)*(yDivision+1+1))
for elemx in range(p,p+xDivision+1):
    for elemy in range(q,q+xDivision+1):
        Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths)
        K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        print(elemx)
        #K[elemx-1:ke_dim+elemx-1,elemy-1:ke_dim+elemy-1] += Ke
        #F[elemx-1:elemx+ke_dim-1] += Fe
        pass
print(K)
print(F)
result = FEM.solve(K,F)
#NURBS.plotNURBSbasisFunction(NControl_u,NControl_w,2,1,weigths,knotvector_u,knotvector_w,p,q,NURBS.R2)
FEM.visualizeResults(Surfacepoints,ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)

#NURBS.plot_surface(Surfacepoints,ctrlpts)

