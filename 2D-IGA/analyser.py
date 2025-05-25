import NURBS
import numpy as np
import FEM
import mesh
from math import sqrt,cos,pi
import matplotlib.pyplot as plt

def loadfunction(x,y):
    #return 2-(x**2 + y**2)
    return - pi**2 /2 *cos(pi*x/2)*cos(pi*y/2)
def solutionfunction(x,y):
    return cos(pi*x/2)*cos(pi*y/2)

test_values = [15,20,40]
esize = [1/(nd+1) for nd in test_values]
orders = [1,2,3]
fig,ax = plt.subplots()
for order in orders:
    accuracy = []
    for division in test_values:
        default = mesh.getDefaultValues(division,order=order)
        x0, y0,x1,y1,xDivision,yDivision,p,q = default
        knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
        assert p==q
        x = np.linspace(x0,x1,10)
        y = np.linspace(y0,y1,7)
        NControl_u = len(knotvector_u)-p-1
        NControl_w = len(knotvector_w)-p-1
        Surfacepoints = []
        """for xx in x:
            srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
            Surfacepoints.append(srf)"""
        K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
        F = np.zeros((xDivision+p+1)*(yDivision+q+1))
        for elemx in range(p,p+xDivision+1):
            for elemy in range(q,q+xDivision+1):
                Ke,Fe = FEM.elemantBspline(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,ctrlpts,loadfunction)
                K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        dirichlet = mesh.getDirichletPoints(int(sqrt(len(F))))
        #print(dirichlet)
        result = FEM.solve(K,F,dirichlet)
        accuracy.append(FEM.calculateErrorBspline(Surfacepoints,ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q,solutionfunction))
    #ax.semilogy(test_values,accuracy)
    ax.loglog(esize,accuracy)
print(accuracy)
ax.set_title("Convergence of MSE based on number of elements")
#ax.set_xlabel("Number of divisions")
ax.set_xlabel("log(Element size)")
ax.set_ylabel("log(Mean Square Error)")
ax.legend(["p=1","p=2","p=3"])

plt.show()