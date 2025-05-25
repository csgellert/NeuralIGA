from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt

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
            Jxi =  0
            Jeta = 0
            #calculating Jxi (dxdxi) and Jeta (dydeta)
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1):
                    dNidxi = dBdXi(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    dNidEta = B(xi, p, xbasisi,knotvector_x)*dBdXi(eta,q,ybasisi,knotvector_y)
                    Jxi += ctrlpts[(nControlx+1)*xbasisi+ybasisi]*dNidxi
                    Jeta += ctrlpts[(nControlx+1)*xbasisi+ybasisi]*dNidEta
            #Calculating the Jacobian
            Jacobi = Jxi*Jeta
            #Jacobi = 1/Jacobi
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            dNidxi = dBdXi(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                            dNidEta = B(xi, p, xbasisi,knotvector_x)*dBdXi(eta,q,ybasisi,knotvector_y)
                            dNjdxi = dBdXi(xi,p,xbasisj,knotvector_x)*B(eta,q,ybasisj,knotvector_y)
                            dNjdeta = B(xi,p,xbasisj,knotvector_x)*dBdXi(eta,q,ybasisj,knotvector_y)
                            
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((dNidxi/Jxi)*(dNjdxi/Jxi) + (dNidEta/Jeta)*(dNjdeta/Jeta) )*Jacobi)
            #* Calculating the Fe vector
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    px = sum([ctrlpts[(nControlx+1)*xbasisi+ybasisi]*B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y) for xbasisi in range(i-p,i+1) for ybasisi in range(j-q,j+1)])
                    py = sum([ctrlpts[(nControlx+1)*xbasisi+ybasisi]*B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y) for xbasisi in range(i-p,i+1) for ybasisi in range(j-q,j+1)])
                    fi = -loadfun(px,py)
                    Ni = B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni*Jacobi)
    return K,F
    
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


def plotAlayticHeatmap(solfun,n=10):
    x_values = np.linspace(0, 1.1, 1000)
    y_values = np.linspace(0, 1.1, 1000)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros((1000,1000))
    # Evaluate the function at each point in the grid
    for idxx, xx in enumerate(x_values):
        for idxy,yy in enumerate(y_values):
            Z[idxy, idxx] = solfun(xx,yy)

    #Z = distanceFromContur(X, Y)

    # Create a contour plot
    plt.contourf(X, Y, Z,levels=100)
    plt.colorbar(label='u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('A modellprobléma megoldása')
    plt.grid(False)
    plt.show()

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