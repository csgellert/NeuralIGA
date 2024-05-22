from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
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
def element(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
    """
    p - order in x direction
    q - order in y direction
    knotvector_x - knotvector in x direction
    knotvector_y - knotvector in y direction
    ed - TODO
    i: ith element in x direction
    j: jth element in y direction
    """
    assert q==p
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    sum = 0
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
            Jxi=1
            Jeta=1
            #Jacobi = 1/Jacobi
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidxi = dR2dXi(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidEta = dR2dEta(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dR2dXi(nControlx,nControly,xbasisj,ybasisj,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdeta = dR2dEta(nControlx,nControly,xbasisj,ybasisj,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((dNidxi/Jxi)*(dNjdxi/Jxi) + (dNidEta/Jeta)*(dNjdeta/Jeta) )*Jacobi)
            #* Calculating the Fe vector
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    px = Surface(nControlx,nControly,xi,eta,weigths,knotvector_x,knotvector_y,p,q,ctrlpts)[0]#xi#knotvector_x[i+xbasisi-i+1]
                    py = Surface(nControlx,nControly,xi,eta,weigths,knotvector_x,knotvector_y,p,q,ctrlpts)[1]#eta#knotvector_y[j+ybasisi-j+1]
                    fi = 2-(px**2 + py**2)
                    Ni = R2(nControlx,nControly,xbasisi,ybasisi,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni*Jacobi)
    return K,F
def elemantBspline(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts,loadfun):
    assert q==p
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    #K = np.zeros((4,4))
    #F = np.zeros(4)
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
            Jxi=1
            Jeta=1
            #Jacobi = 1/Jacobi
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidxi = dBdXi(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                            dNidEta = B(xi, p, xbasisi,knotvector_x)*dBdXi(eta,q,ybasisi,knotvector_y)
                            dNjdxi = dBdXi(xi,p,xbasisj,knotvector_x)*B(eta,q,ybasisj,knotvector_y)
                            dNjdeta = B(xi,p,xbasisj,knotvector_x)*dBdXi(eta,q,ybasisj,knotvector_y)
                            
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((dNidxi/Jxi)*(dNjdxi/Jxi) + (dNidEta/Jeta)*(dNjdeta/Jeta) )*Jacobi)
            #* Calculating the Fe vector
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    px = xi
                    py = eta
                    fi = -loadfun(px,py)
                    Ni = B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni*Jacobi)
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
    
def assembly(K,F,Ke,Fe,elemx,elemy,p,q, xDivision, yDivision):
    l = len(Fe)
    idxs = []
    for idxx in range(p+1):
        for idxy in range(q+1):
            idxs.append((elemx-p)*(xDivision+p+1)+(elemy-q) +idxx*(xDivision+p+1)+idxy)
    #print("idxs:",idxs)
    for idxx,i in enumerate(idxs):
        for idxy,j in enumerate(idxs):
            K[i,j] += Ke[idxx,idxy]
    for idx, i in enumerate(idxs):
        F[i] += Fe[idx]
    #K[elemx-1 + (elemy-1)*3] = K[0,0]

    #K[idx:idx+len(Fe),idx:idx+len(Fe)] = Ke
    return K,F
def solve(K,F,dirichlet):
    k = len(F)
    """
    if k == 16:
        dirichlet = [3,7,11,12,13,14,15]
    elif k == 25:
        dirichlet = [4,9,14,19,20,21,22,23,24]
    elif k == 9:
        dirichlet = [2,5,6,7,8]
    else:
        raise NotImplementedError"""
    mask = np.ones((k, k), dtype=bool)
    for i in dirichlet:
        mask[i,:] = False
        mask[:,i] = False
    #mask[np.ix_(dirichlet, dirichlet)] = False
    tmp = K[mask]
    filtered_K = K[mask].reshape(k - len(dirichlet), k - len(dirichlet))
    mask = np.ones(k, dtype=bool)
    mask[np.ix_(dirichlet)] = False
    filtered_F = F[mask].reshape(k - len(dirichlet))
    
    u = np.dot(np.linalg.inv(filtered_K),filtered_F)
    #print("\nU:\n",u)
    u_orig = u
    for i in dirichlet:
        u_orig = np.insert(u_orig,i,0)
    return u_orig
def visualizeResults(surface, ctrlpts, result,k,l,weigths,knotvector_u,knotvector_w,p,q,calc_error = True):
    
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
            for i in range(0,k):
                for j in range(0,l):
                    res += result[k*i+j]*R2(k,l,i,j,koordinate[0],koordinate[1],weigths,knotvector_u,knotvector_w,p,q)
            zPoints.append(res) 
            zRes.append( 0.5*(koordinate[0]**2-1)*(koordinate[1]**2-1))
    ax.scatter(xPoints,yPoints,zPoints)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,zRes)
    if calc_error:
        MSE = (np.square(np.array(zRes)-np.array(zPoints))).mean()
        print(f"MSE: {MSE}")
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
def visualizeResultsBspline(results,p,q,knotvector_x, knotvector_y,solfun):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(0,1,10)
    y = np.linspace(0,1,10)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            analitical.append(solfun(xx,yy))
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            result.append(sum)
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,analitical)
    
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    plt.show()

def calculateError(surface, ctrlpts, result,k,l,weigths,knotvector_u,knotvector_w,p,q):
    xPoints = []
    yPoints = []
    zPoints = []
    zRes = []
    for surfpoint in surface:
        for koordinate in surfpoint:
            xPoints.append(koordinate[0])  
            yPoints.append(koordinate[1])  
            res  = 0
            for i in range(0,k):
                for j in range(0,l):
                    res += result[k*i+j]*R2(k,l,i,j,koordinate[0],koordinate[1],weigths,knotvector_u,knotvector_w,p,q)
            zPoints.append(res) 
            zRes.append( 0.5*(koordinate[0]**2-1)*(koordinate[1]**2-1))
    if True:
        MSE = (np.square(np.array(zRes)-np.array(zPoints))).mean()
        print(f"MSE: {MSE}")

    return(MSE)
def calculateErrorBspline(surface, ctrlpts, results,k,l,weigths,knotvector_u,knotvector_w,p,q,solfun):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(0,1,10)
    y = np.linspace(0,1,10)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            analitical.append(solfun(xx,yy))
            for xbasis in range(len(knotvector_u)-p-1):
                for ybasis in range(len(knotvector_w)-q-1):
                    sum += B(xx,p,xbasis,knotvector_u)*B(yy,q,ybasis,knotvector_w)*results[(len(knotvector_u)-p-1)*xbasis+ybasis]
            result.append(sum)
    
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    return MSE
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
def RectangleIntegration2D(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
    assert q==p
    #* Defining Gauss points
    g = np.linspace(0,0.25,100)
    dx = g[-1]-g[-2]
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNidxi = dR2dXi(nControlx,nControly,xbasisi,ybasisi,gpx,gpy,weigths,knotvector_x,knotvector_y,p,q)
                            dNidEta = dR2dEta(nControlx,nControly,xbasisi,ybasisi,gpx,gpy,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dR2dXi(nControlx,nControly,xbasisj,ybasisj,gpx,gpy,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdeta = dR2dEta(nControlx,nControly,xbasisj,ybasisj,gpx,gpy,weigths,knotvector_x,knotvector_y,p,q)
                            
                            K[(p+1)*(xbasisi-i+1) + ybasisi-j+1][(p+1)*(xbasisj-i+1)+(ybasisj-j+1)] += (((dNidxi)*(dNjdxi) + (dNidEta)*(dNjdeta)))*dx*dx
    return K


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
    iGaLA = gaussLagandereQuadratureBasisfunction(i,t,k,1)
    print(f"Rectangle: {iRe}\tGauss: {iGaLA}")
     #element(2,2,t,t,None,1,1)
    plt.show()