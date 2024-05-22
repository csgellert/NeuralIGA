import numpy as np
from NURBS import Surface, plot_surface
def generateRectangularMesh(x0, y0, x1, y1, xDivision,yDivision,p=1,q=1):
    assert x0 < x1 and y0 < y1
    knotvector_u = np.linspace(x0,x1,xDivision+2)
    knotvector_w = np.linspace(y0,y1,yDivision+2)
    weights = np.ones((yDivision+q+1,xDivision+p+1))
    ctrlpts = []
    controlx = np.linspace(x0,x1,xDivision+p+1)
    controly = np.linspace(y0,y1,yDivision+q+1)
    for j in controly:
        row = []
        for i in controlx:
            point = [i, j,1]
            row.append(point)
        ctrlpts.append(row)

    knotvector_u = np.insert(knotvector_u,0,[x0 for _ in range(p)])
    knotvector_u = np.append(knotvector_u,[x1 for _ in range(p)])
    knotvector_w = np.insert(knotvector_w,0,[y0 for _ in range(q)])
    knotvector_w = np.append(knotvector_w,[y1 for _ in range(q)])
    return knotvector_u, knotvector_w, weights, ctrlpts
def getDefaultValues(div=2,order=1):
    x0 = 0
    x1 = 1
    y0 = 0
    y1 = 1
    p = order
    q = order
    xDivision = div
    yDivision = div
    return x0, y0,x1,y1,xDivision,yDivision,p,q
def getDirichletPoints(k):
    dirichlet = [i*k+k-1 for i in range(k-1)]
    for i in range(k):
        dirichlet.append(i+k*(k-1))
    #print(f"Dirichlet boundary: {dirichlet}")
    return dirichlet
    
if __name__ == "__main__":
    x0 = 0
    x1 = 2
    y0 = -3
    y1 = 1
    p = 3
    q = 3
    xDivision = 2
    yDivision = 2
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
