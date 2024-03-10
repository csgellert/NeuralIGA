import numpy as np
from Geomertry import B
import matplotlib.pyplot as plt
import math
GY = math.sqrt(2)/2
PP2 = math.pi/2

#circle
order = 2
ctrlpts=   [[1,0,1],
             [1,1,1],
             [0,1,1],
             [-1,1,1],
             [-1,0,1],
             [-1,-1,1],
             [0,-1,1],
             [1,-1,1],
             [1,0,1]]
weigths = [1,GY,1,GY,1,GY,1,GY,1]
knotvector = [0,0,0,PP2,PP2,2*PP2,2*PP2,3*PP2,3*PP2,4*PP2,4*PP2,4*PP2]
x = np.linspace(0,PP2*4,100)
NControl = 9


def get_denominator(k,u,weigths, knotvector,order):
    sum = 0
    # amit a  Wikipédia u-nak jelöl az az x a mi esetünkben...
    for i in range(1-1,k):
        sum += weigths[i]*B(u,order,i,knotvector)
    return sum
def get_nominator(k,u,weigths, knotvector,order,ctrlpts):
    sum = np.array([0,0,0],dtype='float64')
    # amit a  Wikipédia u-nak jelöl az az x a mi esetünkben...
    for i in range(1-1,k):
        Btmp = B(u,order,i,knotvector)
        ctemp = ctrlpts[i]
        wtmp = weigths[i]
        sum += weigths[i]*B(u,order,i,knotvector)*np.array(ctrlpts[i])
    return list(sum)

def Curve(k,u,weigths, knotvector,order,ctrlpts):
    C = get_nominator(k,u,weigths, knotvector,order,ctrlpts)/get_denominator(k,u,weigths, knotvector,order)
    return C

def plotcurve(curvepoints,ctrlpoints,weigths):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x = [x[0] for x in curvepoints]
    y = [x[1] for x in curvepoints]
    z = [x[2] for x in curvepoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')
    ax.scatter(x,y,z)

    #plot controlpoints
    x = [x[0] for x in ctrlpoints]
    y = [x[1] for x in ctrlpoints]
    z = [x[2] for x in ctrlpoints]
    ax.scatter(x,y,z,c="g")
    #plot weigthed controlpoints
    x = [x[0]*weigths[idx] for idx,x in enumerate(ctrlpoints)]
    y = [x[1]*weigths[idx] for idx,x in enumerate(ctrlpoints)]
    z = [x[2]*weigths[idx] for idx,x in enumerate(ctrlpoints)]
    ax.scatter(x,y,z,c="r",marker="*")
    
    plt.show()


Curvepoints = [Curve(NControl,xx,weigths,knotvector,order,ctrlpts) for xx in x]
#print(Curvepoints)
plotcurve(Curvepoints,ctrlpts,weigths)
