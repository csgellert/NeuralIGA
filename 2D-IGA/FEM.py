from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
def shapeFunctionRoutine(k, u, p, q, knotvector_u, knotvector_v,ctrlpts,weigths, nen,e, INN, IEN, xi_tilde,eta_tilde):
    """
    p: order in x direction
    q: order in y deirection
    nen: number of the local shape function
    e: element number
    INC: TODO
    ICN: TODO
    xi_tilde: TODO
    eta_tilde: TODO
    """
    #* --------------- Initialisation-----------------
    R = np.zeros(nen) #Array of the trivariate NURBS basis functions
    dR_dx = np.zeros((nen,2)) # Bivariate NURBS functon derivatives w.r.t. physical coordinates
    J = 0

    #*Local variable initialization
    ni,nj=0 # NURBS coordiates
    xi, eta = 0 #Parametric coordinates
    N = np.zeros(p+1) # Arrays of uninvariant B-spline basis functions
    M = np.zeros(q+1) # Arrays of uninvariant B-spline basis functions    

    dN_dxi = np.zeros(p+1) #Uninvariant B-spline function derivatives w.r.t appropriete parametric coordinates
    dM_deta = np.zeros(q+1) #Uninvariant B-spline function derivatives w.r.t appropriete parametric coordinates
    dR_dxi = np.zeros((nen,2)) #Bivariante NURBS function derivatives w.r.t parametric coordinates

    dx_dxi = np.zeros((2,2)) # Derivative od parametric coordinates w.r.t parametric coordinates
    
    dxi_dx = np.zeros((2,2)) # Inverse of dx dxi
    dxi_dtildexi = np.zeros((2,2)) #Derivatives of parametric coordinates w.r.t. parent element coordinates

    J_mat = np.zeros((2,2)) #Jacobian matrix
    #?counters
    sum_xi,sum_eta = 0 #Dummy sums for calculating rational derivatives

    # NURBS coordinates
    ni = INN[[IEN[e][0]]][0]
    nj = INN[[IEN[e][0]]][1]

    xi = ((knotvector_u[ni+1]-knotvector_u[ni])*xi_tilde + (knotvector_u[ni+1]+knotvector_u[ni]))/2
    eta = ((knotvector_w[nj+1]-knotvector_w[nj])*eta_tilde + (knotvector_w[nj+1]+knotvector_w[nj]))/2

    #*--------------------Part 2 -----------------------------------------------------------
    #?BsplineBasis and Derivatives:
    #TODO
def gaussIntegrateElement(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths):
    """
    p - order in x direction
    q - order in y direction
    knotvector_x - knotvector in x direction
    knotvector_y - knotvector in y direction
    ed - TODO
    i: ith element in x direction
    j: jth element in y direction
    """
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    sum = 0

    for xbasis in range(0,p+1):#iterate throug basis functions in x direction
        for ybasis in range(0,q+1): #iterate throug basis functions in y direction
            for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
                for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
                    xi = (x2-x1)/2 * gpx + (x2+x1)/2
                    eta = (y2-y1)/2 * gpy + (y2+y1)/2
                    Jxi = (x2-x1)/2
                    Jeta = (y2-y1)/2
                    f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                    sum += Jxi*Jeta*w[idxx]*w[idxy]*f
    return sum
def element(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths):
    """
    p - order in x direction
    q - order in y direction
    knotvector_x - knotvector in x direction
    knotvector_y - knotvector in y direction
    ed - TODO
    i: ith element in x direction
    j: jth element in y direction
    """
    assert q==1 and p == 1 
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    sum = 0
    K = np.zeros((4,4))
    F = np.zeros(4)
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
            Jxi=1
            Jeta=1
            #*CAlculating the Ke
            for xbasisi in range(0,2):
                for ybasisi in range(0,2): 
                    for xbasisj in range(0,2):
                        for ybasisj in range(0,2): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidxi = dR2dXi(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidEta = dR2dEta(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dR2dXi(nControlx,nControly,xbasisj,ybasisj,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdeta = dR2dEta(nControlx,nControly,xbasisj,ybasisj,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            #raise NotImplementedError #! Ezt itt még nagyon át kéne nézni, de lehet nem este 10 0 legalkalmasabb erre...
                            K[2*xbasisi + ybasisi][2*xbasisj+ybasisj] += w[idxx]*w[idxy]*(((dNidxi/Jxi)*(dNjdxi/Jxi) + (dNidEta/Jeta)*(dNjdeta/Jeta) )*Jacobi)
            #* Calculating the Fe vector
            for xbasisi in range(0,2):
                for ybasisi in range(0,2): 
                    fi = 2-(xi**2 + eta**2)
                    Ni = R2(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                    F[2*xbasisi+ybasisi] += w[idxx]*w[idxy]*(fi*Ni*Jacobi)
    if x2 == 1:
        K[1:2][:] = 0
        K[:][1:2] = 0
    if y2 == 1:
        K[2:3][:] = 0
        K[:][2:3] = 0
    return K,F

def integrateElement(k,l,weigths,knotvector_u,knotvector_w,p,q):
    x = np.linspace(0,2,100)
    y = np.linspace(0,3,100)
    dx = x[-1]-x[-2]
    dy = y[-1]-y[-2]
    sum = 0
    for xx in x:
        for yy in y:
            for i in range(0,k):
                for j in range(0,l):
                    sum+=R2(k,l,i,j,xx,yy,weigths,knotvector_u,knotvector_w,p,q)*dx*dy
    return sum

    
    
def assembly(K,F,Ke,Fe):
    pass
def solve(K,F):
    u = np.dot(np.linalg.inv(K),F)
    print("U:\n",u)
    return u
def visualizeResults(surface, ctrlpts, result,k,l,weigths,knotvector_u,knotvector_w,p,q):
    
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    xPoints = []
    yPoints = []
    zPoints = []
    zRes = []
    #result.append([0,0,0,0])
    #plotBsplineBasis(np.linspace(0,1,100), knotvector_u,p)
    for surfpoint in surface:
        for koordinate in surfpoint:
            xPoints.append(koordinate[0])  
            yPoints.append(koordinate[1])  
            res  = 0
            for i in range(0,2):
                for j in range(0,2):
                    res += result[2*j+i]*R2(k,l,i,j,koordinate[0],koordinate[1],weigths,knotvector_u,knotvector_w,p,q)
            zPoints.append(res) 
            zRes.append( 0.5*(koordinate[0]**2-1)*(koordinate[1]**2-1))
    ax.scatter(xPoints,yPoints,zPoints)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,zRes)
    #plot controlpoints:
    x=[]
    y=[]
    z=[]
    for j in ctrlpts:
        for i in j:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
    #ax.scatter(x,y,z,c="r",marker="*")
    #plt.axis('equal')
    plt.show()
def gaussLagandereQuadratureBasisfunction(i,knotvector, order, gaussPoints=1, func=B):
    if gaussPoints == 1:
        g = [-1/math.sqrt(3), 1/math.sqrt(3)]
        w = [1,1]
    elif gaussPoints == 2:
        g = [-math.sqrt(15)/5, 0, math.sqrt(15)/5]
        w = [5/9, 8/9, 5/9]
    else:
        raise NotImplementedError
    sum = 0
    for element in range(i,i+order+2):
        a = knotvector[element]
        b = knotvector[element+1]
        difp2 = (b-a)/2
        avg = (a+b)/2
        pass
        for idx,gaussPoint in enumerate(g):
            xi = difp2*gaussPoint+avg
            sum += difp2* w[idx]*func(xi, order,i,knotvector)
    return sum
def RectangleIntegration(knotvector, i, order, division,func=B):
    x = np.linspace(knotvector[0],knotvector[-1], division)
    dx = x[-1]-x[-2]
    sum = 0
    for xx in x:
        sum += func(xx,order,i,knotvector)*dx
    return sum



#* TEST
if __name__ == "__main__":
    k = 3

    import numpy as np
    fig, ax = plt.subplots()
    xx2 = np.linspace(-3, 3, 500)
    t = [-3,-3,-3,-3,0,1,1, 3,3,3,3]
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
    i = 3
    iRe = RectangleIntegration(t,i,k,50000)
    iGaLA = gaussLagandereQuadratureBasisfunction(i,t,k,2)
    print(f"Rectangle: {iRe}\tGauss: {iGaLA}")
     #element(2,2,t,t,None,1,1)
    plt.show()