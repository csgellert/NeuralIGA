import numpy as np
from Geomertry import B
import matplotlib.pyplot as plt
import math
GY = math.sqrt(2)/2
PP2 = math.pi/2



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
def get_denominator2d(k,l,u,w,weigths, knotvector_u,knotvector_w,p,q):
    sum = 0
    # amit a  Wikipédia u-nak jelöl az az x a mi esetünkben...
    for i in range(1-1,k):
        for j in range(1-1,l):
            tmpB1 = B(u,p,i,knotvector_u)
            tmpB2 = B(w,q,j,knotvector_w)
            tmpw = weigths[j][i]
            sum += weigths[j][i]*B(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)
    if sum == 0:
        sum=1
    return sum

def get_nominator2d(k,l,u,w,weigths, knotvector_u,knotvector_w,p,q,ctrlpts):
    sum = np.array([0,0,0],dtype='float64')
    # amit a  Wikipédia u-nak jelöl az az x a mi esetünkben...
    for i in range(1-1,k):
        for j in range(1-1,l):
            tmpw = weigths[j][i]
            tmp_c = ctrlpts[j][i]
            sum += weigths[j][i]*B(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)*np.array(ctrlpts[j][i])
    return list(sum)

def Curve(k,u,weigths, knotvector,order,ctrlpts):
    C = get_nominator(k,u,weigths, knotvector,order,ctrlpts)/get_denominator(k,u,weigths, knotvector,order)
    return C

def Surface(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q,ctrlpts):
    n = get_nominator2d(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q,ctrlpts)
    d = get_denominator2d(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)
    tmp = n/d
    S = get_nominator2d(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q,ctrlpts)/get_denominator2d(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)
    if math.isnan(S[0]):
        S = 0
    return S


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
def plot_surface(surface):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    xPoints = []
    yPoints = []
    zPoints = []
    for i in surface:
        for q in i:
            xPoints.append(q[0])  
            yPoints.append(q[1])  
            zPoints.append(q[2]) 
    # xPoints = [item for item in xPoints if not(math.isnan(item)) == True]
    # yPoints = [item for item in yPoints if not(math.isnan(item)) == True]
    # zPoints = [item for item in zPoints if not(math.isnan(item)) == True]

    #ax.plot_surface(xBezier,yBezier,zBezier)
    ax.scatter(xPoints,yPoints,zPoints)#, edgecolors='face')
    plt.show()



surface = True
if __name__== "__main__":
    if not surface:
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
        Curvepoints = [Curve(NControl,xx,weigths,knotvector,order,ctrlpts) for xx in x]
        #print(Curvepoints)
        plotcurve(Curvepoints,ctrlpts,weigths)
    else:
        p = 2
        q = 1
        ctrlpts=   [[[1,0,1],
                    [1,1,1],
                    [0,1,1],
                    [-1,1,1],
                    [-1,0,1],
                    [-1,-1,1],
                    [0,-1,1],
                    [1,-1,1],
                    [1,0,1]],
                    [[2,0,1],
                    [2,2,1],
                    [0,2,1],
                    [-2,2,1],
                    [-2,0,1],
                    [-2,-2,1],
                    [0,-2,1],
                    [2,-2,1],
                    [2,0,1]]]
        weigths = [[1,GY,1,GY,1,GY,1,GY,1],
                   [1,GY,1,GY,1,GY,1,GY,1]]
        knotvector_u = [0,0,0,PP2,PP2,2*PP2,2*PP2,3*PP2,3*PP2,4*PP2,4*PP2,4*PP2]
        knotvector_w = [1,1,2,2]
        phi = np.linspace(0,PP2*4-1e-5,80)
        r = np.linspace(1,2-1e-5,10)
        NControl_u = 9
        NControl_w = 2
        Surfacepoints = []
        for ph in phi:
            srf = [Surface(NControl_u,NControl_w,ph,rr,weigths,knotvector_u,knotvector_w,p,q,ctrlpts) for rr in r]
            Surfacepoints.append(srf)
        #print(Curvepoints)
        print(Surfacepoints)
        plot_surface(Surfacepoints)