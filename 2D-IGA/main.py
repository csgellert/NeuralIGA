import NURBS
import numpy as np
import FEM

#defining geometry:
p = 1
q = 1
ctrlpts=   [[[0,0,1],
            [0,1,1],
            ],
            [[1,0,1],
             [1,1,1]
            ],
            [[2,0,1],
            [2,1,1],
            ]]
weigths = [[1,1],
           [1,1],
           [1,1]]
ctrlpts_simple = [[[0,0,1],
            [0,1,1],
            ],
            [[1,0,1],
            [1,1,1],
            ]]
weigths = [[1,1],
           [1,1]]
knotvector_u = [0,0,2,2] #x
knotvector_w = [0,0,3,3] #y
x = np.linspace(0,2,10)
y = np.linspace(0,3,7)
NControl_u = 2
NControl_w = 2

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts_simple) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
#print(Surfacepoints)
iGa = FEM.gaussIntegrateElement(p,q,knotvector_u,knotvector_w,None,1,1,NControl_u,NControl_w,weigths)
iRe = FEM.integrateElement(NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
K,F = FEM.element(p,q,knotvector_u,knotvector_w,None,1,1,NControl_u,NControl_w,weigths)
print(K)
print(np.linalg.inv(K))
print(F)
result = FEM.solve(K,F)
FEM.visualizeResults(Surfacepoints,ctrlpts_simple,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
print(f"Gauss: {iGa}\tRectangle: {iRe}")
NURBS.plot_surface(Surfacepoints,ctrlpts_simple)

