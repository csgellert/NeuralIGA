import numpy as np
from NURBS import Surface, plot_surface
from math import sqrt
import matplotlib.pyplot as plt
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
def getDefaultValues(div=2,order=1,delta = 0):
    assert delta >=0
    x0 = -1-delta
    x1 = 1+delta
    y0 = -1-delta
    y1 = 1+delta
    p = order
    q = order
    xDivision = div
    yDivision = div
    return x0, y0,x1,y1,xDivision,yDivision,p,q
def distanceFromContur(x,y):
    #Circle like domain
    # Inside of the domain is negative so it is better compatible with the sigmoid function
    R2 = 1*1
    r = x**2 + y**2
    return R2-r
def dddx(x,y):
    return 2*x
def dddy(x,y):
    return 2*y
def plotMesh(xdiv=2, ydiv=3,delta=0):
    circle = plt.Circle((0, 0), 1, color='r')
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    ax.set_xticks(np.arange(-1-delta, 1+delta, (2+2*delta)/(xdiv+1)))
    ax.set_yticks(np.arange(-1-delta, 1+delta, (2+2*delta)/(ydiv+1)))
    ax.set_xlim((-1-delta,1+delta))
    ax.set_ylim((-1-delta,1+delta))
    ax.set_aspect('equal')
    
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    plotMesh(1,1,0.1)
    x0 = -1
    x1 = 1
    y0 = -1
    y1 = 1
    p = 1
    q = 1
    xDivision = 5
    yDivision = 5
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
