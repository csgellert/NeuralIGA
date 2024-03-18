from NURBS import *
from Geomertry import *
import numpy as np

def element(rr, ph, p, q, knotvector_u, knotvector_v,ctrlpts,weigths):
    pass
def assembly(K,F,Ke,Fe):
    pass
def solve(K,F):
    pass
def GaussLagrandeQuadrature(i,knotvector, order, gaussPoints=1):
    if gaussPoints == 1:
        g = [-1/math.sqrt(3), 1/math.sqrt(3)]
        w = [1,1]
    elif gaussPoints == 2:
        g = [-math.sqrt(15)/5, 0, math.sqrt(15)/5]
        w = [5/9, 8/9, 5/9]
    else:
        raise NotImplementedError
    #numerical integration
    sum = 0
    for idx,gaussPoint in enumerate(g):
        sum += w[idx]*B(gaussPoint, order,i,knotvector)
    #TODO Jacobi determinant!!!
    return sum
def RectangleIntegration(knotvector, i, order, division):
    x = np.linspace(knotvector[0],knotvector[-1], division)
    dx = x[-1]-x[-2]
    sum = 0
    for xx in x:
        sum += B(xx,order,i,knotvector)*dx
    return sum



#* TEST
if __name__ == "__main__":
    k = 2

    import numpy as np
    fig, ax = plt.subplots()
    xx2 = np.linspace(-1, 1, 500)
    t = [-1,-1,-1,0, 1,1,1]
    c = [1,1,1,1,1,1,1,1,1,1,1,1]
    ax.plot(xx2, [bspline(x, t, c ,k) for x in xx2], 'g-', lw=3, label='naive')
    ax.grid(True)
    ax.legend(loc='best')

    for i in range(len(t)-k-1):
        #if i == 0 or i== len(t)-k-1-1: continue
        #Ni = [B(x,k,i,t) for x in xx]
        Ni2 = [B(x,k,i,t) for x in xx2]
        #diff = [(Ni2[idx]-Ni2[idx-1])/(xx2[1]-xx2[0]) for idx,x in enumerate(Ni2)]
        #ax.plot(xx, Ni)2
        ax.plot(xx2, Ni2)
        ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx2],"g--")
        ax.plot(t,[0 for _ in t],"r*")
        #ax.plot(xx2[1:], diff[1:])
    #* TEST Integration
    iLa = GaussLagrandeQuadrature(1,t,k,1)
    iRe = RectangleIntegration(t,1,k,10000)
    print(f"Gauss: {iLa}\tRectangle: {iRe}")
    plt.show()