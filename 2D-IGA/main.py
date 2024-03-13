import NURBS
import numpy as np

#defining geometry:
p = 1
q = 2
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
knotvector_u = [0,0,2,2] #x
knotvector_w = [0,0,0,1,1,1] #y
x = np.linspace(0,2-1e-3,10)
y = np.linspace(0,1-1e-3,7)
NControl_u = 2
NControl_w = 3

Surfacepoints = []
for xx in x:
    srf = [NURBS.Surface(NControl_u,NControl_w,xx,yy,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for yy in y]
    Surfacepoints.append(srf)
#print(Curvepoints)
print(Surfacepoints)
NURBS.plot_surface(Surfacepoints,ctrlpts)

