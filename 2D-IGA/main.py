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
ctrlpts2x2 = [[[0,0,1],
               [0,0.5,1],
            [0,1,1],
            ],
            [[0.5,0,1],
             [0.5,0.5,1],
             [0.5,1,1]
            ],
            [[1,0,1],
             [1,0.5,1],
            [1,1,1],
            ]]
weigths = [[1,1],
           [1,1]]
weigths = [[1,1,1],
           [1,1,1],
           [1,1,1]]
knotvector_u = [0,0,0.5,1,1] #x
knotvector_w = [0,0,0.5,1,1] #y
x = np.linspace(0,1,10)
y = np.linspace(0,1,7)
NControl_u = 3
NControl_w = 3

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts2x2) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
#print(Surfacepoints)
iGa = FEM.gaussIntegrateElement(p,q,knotvector_u,knotvector_w,None,1,1,NControl_u,NControl_w,weigths)
iRe = FEM.integrateElement(NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)

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
FEM.visualizeResults(Surfacepoints,ctrlpts_simple,result,NControl_u,NControl_w,weigths,knotvector_u,knotvector_w,p,q)
print(f"Gauss: {iGa}\tRectangle: {iRe}")
NURBS.plot_surface(Surfacepoints,ctrlpts2x2)

