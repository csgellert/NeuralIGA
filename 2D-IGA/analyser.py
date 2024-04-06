import NURBS
import numpy as np
import FEM
import mesh
from math import sqrt
import matplotlib.pyplot as plt

test_values = [0,1,2,3,4,5]
accuracy = []
for division in test_values:
    default = mesh.getDefaultValues(division,division)
    x0, y0,x1,y1,xDivision,yDivision,p,q = default
    knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
    assert p==1 and q==1
    x = np.linspace(x0,x1,10)
    y = np.linspace(y0,y1,7)
    NControl_u = len(knotvector_u)-p-1
    NControl_w = len(knotvector_w)-p-1
    Surfacepoints = []
    for xx in x:
        srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
        Surfacepoints.append(srf)
    K = np.zeros(((xDivision+1+1)*(yDivision+1+1),(xDivision+1+1)*(yDivision+1+1)))
    F = np.zeros((xDivision+1+1)*(yDivision+1+1))
    for elemx in range(p,p+xDivision+1):
        for elemy in range(q,q+xDivision+1):
            Ke,Fe = FEM.element(p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths)
            K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
    dirichlet = mesh.getDirichletPoints(int(sqrt(len(F))))
    print(dirichlet)
    result = FEM.solve(K,F,dirichlet)
    accuracy.append(FEM.calculateError(Surfacepoints,ctrlpts,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q))
fig,ax = plt.subplots()
ax.semilogy(test_values,accuracy)
ax.set_title("Convergence of MSE based on number of elements")
ax.set_xlabel("Number of divisions")
ax.set_ylabel("log(Mean Square Error)")

plt.show()