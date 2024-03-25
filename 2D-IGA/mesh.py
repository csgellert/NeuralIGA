import numpy as np
from NURBS import Surface, plot_surface
def generateRectangularMesh(x0, y0, x1, y1, xDivision,yDivision,p=1,q=1):
    assert x0 < x1 and y0 < y1
    knotvector_u = np.linspace(x0,x1,xDivision+2)
    knotvector_w = np.linspace(y0,y1,yDivision+2)
    weights = np.ones((yDivision+2,xDivision+2))
    ctrlpts = []
    for j in range(yDivision+2):
        row = []
        for i in range(xDivision+2):
            point = [knotvector_u[i], knotvector_w[j],1]
            row.append(point)
        ctrlpts.append(row)

    knotvector_u = np.insert(knotvector_u,0,[x0 for _ in range(p)])
    knotvector_u = np.append(knotvector_u,[x1 for _ in range(p)])
    knotvector_w = np.insert(knotvector_w,0,[y0 for _ in range(q)])
    knotvector_w = np.append(knotvector_w,[y1 for _ in range(q)])
    return knotvector_u, knotvector_w, weights, ctrlpts
def getDefaultValues():
    x0 = 0
    x1 = 1
    y0 = 0
    y1 = 1
    p = 1
    q = 1
    xDivision = 1
    yDivision = 1
    return x0, y0,x1,y1,xDivision,yDivision,p,q
if __name__ == "__main__":
    x0 = 0
    x1 = 2
    y0 = -3
    y1 = 1
    p = 1
    q = 1
    xDivision = 2
    yDivision = 3
    knotvector_u, knotvector_w, weights, ctrlpts = generateRectangularMesh(x0,y0,x1,y1,xDivision,yDivision,p,q)
    k = len(knotvector_u)-p-1
    l = len(knotvector_w)-q-1
    y = np.linspace(y0,y1,10)
    x = np.linspace(x0,x1,10)
    surfacepoints = []
    for yy in y:
        surf = [Surface(k,l,xx,yy,weights,knotvector_u,knotvector_w,p,q,ctrlpts) for xx in x]
        surfacepoints.append(surf)
    plot_surface(surfacepoints,ctrlpts)
