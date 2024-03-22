import numpy as np
from Geomertry import B, dBdXi, plotBsplineBasis
import matplotlib.pyplot as plt
import math

def Surface(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q,ctrlpts):
    sum = np.array([0,0,0],dtype='float64')
    for i in range(0,k):
        for j in range(0,l):
            sum += R2(k,l,i,j,u,w,weigths,knotvector_u,knotvector_w,p,q)*np.array(ctrlpts[j][i])
    return list(sum)
def Curve(k,u,weigths, knotvector,order,ctrlpts):
    sum = 0
    for i in range(k):
        sum += R(k,u,i,weigths, knotvector,order)*np.array(ctrlpts[i])
    return sum
def CurveDerivative(k,u,weigths, knotvector,order,ctrlpts):
    sum = 0
    for i in range(k):
        sum += dRdXi(k,u,i,weigths, knotvector,order)*np.array(ctrlpts[i])
    return sum
def W(k,u,weigths, knotvector,order):
    W_sum = 0
    for i in range(k):
        W_sum += weigths[i]*B(u,order,i,knotvector)
    return W_sum
def W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q):
    sum = 0
    for i in range(1-1,k):
        for j in range(1-1,l):
            sum += weigths[j][i]*B(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)
    return sum
def dWdXi(k,u,weigths, knotvector,order):
    W_sum = 0
    for i in range(k):
        W_sum += weigths[i]*dBdXi(u,order,i,knotvector)
    return W_sum
def R(k,u,i,weigths, knotvector,order):
    return weigths[i]*B(u,order,i,knotvector)/W(k,u,weigths, knotvector,order)
def R2(k,l,i,j,u,w,weigths,knotvector_u,knotvector_w,p,q):
    return weigths[j][i]*B(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)/W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)
def dRdXi(k,u,i,weigths, knotvector,order):
    if u == knotvector[-1]:
        print(":::")
    numerator = W(k,u,weigths, knotvector,order)*dBdXi(u,order,i,knotvector) - dWdXi(k,u,weigths, knotvector,order)*B(u,order,i,knotvector)
    denominator = W(k,u,weigths, knotvector,order)*W(k,u,weigths, knotvector,order)
    return weigths[i]*numerator/denominator
def dW2dXi(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q):
    #xi == u 
    sum = 0
    for i in range(1-1,k):
        for j in range(1-1,l):
            sum += weigths[j][i]*dBdXi(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)
    return sum
def dW2dEta(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q):
    #eta = w
    sum = 0
    for i in range(1-1,k):
        for j in range(1-1,l):
            sum += weigths[j][i]*dBdXi(u,p,i,knotvector_u)*dBdXi(w,q,j,knotvector_w)
    return sum
def dR2dXi(k,l,i,j,u,w,weigths,knotvector_u,knotvector_w,p,q):
    numerator = W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*dBdXi(u,p,i,knotvector_u) - dW2dXi(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*B(u,p,i,knotvector_u)
    denominator = W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)
    return weigths[j][i]*B(w,q,j,knotvector_w)*numerator/denominator
def dR2dEta(k,l,i,j,u,w,weigths,knotvector_u,knotvector_w,p,q):
    numerator = W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*dBdXi(w,q,j,knotvector_w) - dW2dEta(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*B(w,q,j,knotvector_w)
    denominator = W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)*W2(k,l,u,w,weigths,knotvector_u,knotvector_w,p,q)
    return weigths[j][i]*B(u,p,i,knotvector_u)*numerator/denominator
def plotNURBSbasisFunction(k,l,i,j,weigths,knotvector_u,knotvector_w,p,q,fun):
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    xPoints = []
    yPoints = []
    zPoints = []
    for xx in x:
        for yy in y:
            xPoints.append(xx)  
            yPoints.append(yy)  
            f = fun(k,l,i,j,xx,yy,weigths,knotvector_u,knotvector_w,p,q)
            zPoints.append(f)
    ax.scatter(xPoints,yPoints,zPoints)
    plt.xlabel("x")
    plt.show()
    
    #plot it


    

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
def plot_surface(surface,ctrlpts):
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
    #plot controlpoints:
    x=[]
    y=[]
    z=[]
    for j in ctrlpts:
        for i in j:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
    ax.scatter(x,y,z,c="r",marker="*")
    plt.axis('equal')
    plt.show()



surface = True
if __name__== "__main__":
    GY = math.sqrt(2)/2
    PP2 = math.pi/2
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
        x = np.linspace(0,PP2*4,1000)
        NControl = 9
        Curvepoints = [Curve(NControl,xx,weigths,knotvector,order,ctrlpts) for xx in x]
        derivative = [CurveDerivative(NControl,xx,weigths,knotvector,order,ctrlpts) for xx in x]
        #print(Curvepoints)
        #plotcurve(Curvepoints,ctrlpts,weigths)
        plotBsplineBasis(x,knotvector,order,derivative=True, sum=False)
        plotcurve(derivative,ctrlpts,weigths)
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
        plot_surface(Surfacepoints, ctrlpts)