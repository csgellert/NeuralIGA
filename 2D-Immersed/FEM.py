from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x*15))
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
def elemantBspline(p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
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
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
            d = mesh.distanceFromContur(xi,eta)
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
                            Ni = B(eta,q,ybasisi,knotvector_y)*B(xi,q,xbasisi,knotvector_x)
                            #correction with the distance function
                            diCorrXi = dNidxi*d + mesh.dddx(xi,eta) * Ni
                            diCorrEta = dNidEta*d + mesh.dddy(xi,eta) * Ni
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((diCorrXi)*(dNjdxi) + (diCorrEta)*(dNjdeta))*Jacobi)
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    px = xi
                    py = eta
                    fi = 2-(px**2 + py**2)
                    Ni = B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni*Jacobi)
    return K,F
def boundaryElementBspline(r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly):
    assert q==p
    SUBDIVISION = 1
    DOSUBDIV = True
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    #doing subdivision
    if DOSUBDIV:
        halfx = (x1+x2)/2
        halfy = (y1+y2)/2
        #first
        Ks,Fs = GaussQuadrature(x1, halfx,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks
        F+=Fs
        #second
        Ks,Fs = GaussQuadrature(halfx, x2,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks
        F+=Fs
        #third
        Ks,Fs = GaussQuadrature(x1, halfx,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks
        F+=Fs
        #fourth
        Ks,Fs = GaussQuadrature(halfx, x2,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks
        F+=Fs
    
    return K,F
def GaussQuadrature(x1,x2,y1,y2,r,i,j,p,q,knotvector_x,knotvector_y):
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            d = mesh.distanceFromContur(xi,eta)
            if d<0: continue
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
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
                            
                            Ni = B(eta,q,ybasisi,knotvector_y)*B(xi,q,xbasisi,knotvector_x)
                            #correction with the distance function
                            diCorrXi = dNidxi*d + mesh.dddx(xi,eta) * Ni
                            diCorrEta = dNidEta*d + mesh.dddy(xi,eta) * Ni
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((diCorrXi)*(dNjdxi) + (diCorrEta)*(dNjdeta))*Jacobi)
            #* Calculating the Fe vector
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    px = xi
                    py = eta
                    fi = 2-(px**2 + py**2)
                    Ni = B(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += (w[idxx]*w[idxy]*(fi*Ni*Jacobi))
    return K,F
def elementChoose(Nurbs_fun,r,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
    assert not Nurbs_fun
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]
    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point<r:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        Ke, Fe = elemantBspline(p,q,knotvector_x, knotvector_y, None,i,j,nControlx, nControly,weigths,ctrlpts)
    elif outerElement:
        Ke = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
        Fe = np.zeros((p+1)*(q+1))
    else:
        Ke, Fe = boundaryElementBspline(r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly)
    return Ke, Fe
    

def assembly(K,F,Ke,Fe,elemx,elemy,p,q, xDivision, yDivision):
    l = len(Fe)
    idxs = []
    for idxx in range(p+1):
        for idxy in range(q+1):
            idxs.append((elemx-p)*(xDivision+p+1)+(elemy-q) +idxx*(xDivision+p+1)+idxy)
    for idxx,i in enumerate(idxs):
        for idxy,j in enumerate(idxs):
            K[i,j] += Ke[idxx,idxy]
    for idx, i in enumerate(idxs):
        F[i] += Fe[idx]
    return K,F
def solve(K,F,dirichlet):
    k = len(F)
    mask = np.ones((k, k), dtype=bool)
    for i in dirichlet:
        mask[i,:] = False
        mask[:,i] = False

    filtered_K = K[mask].reshape(k - len(dirichlet), k - len(dirichlet))
    mask = np.ones(k, dtype=bool)
    mask[np.ix_(dirichlet)] = False
    filtered_F = F[mask].reshape(k - len(dirichlet))
    
    u = np.dot(np.linalg.inv(filtered_K),filtered_F)
    u_orig = u
    for i in dirichlet:
        u_orig = np.insert(u_orig,i,0)
    return u_orig
def solveWeak(K,F):
    zero_rows = np.all(K == 0, axis=1)
    zero_cols = np.all(K == 0, axis=0)
    zero_f = zero_rows
    # Remove zero rows and columns
    K_reduced = K[~zero_rows][:, ~zero_cols]
    F_reduced = F[~zero_f]
    u = np.zeros(len(F))
    u_reduced = np.dot(np.linalg.inv(K_reduced),F_reduced)
    u[~zero_f] = u_reduced
    return u
def visualizeResults(surface, ctrlpts, result,k,l,weigths,knotvector_u,knotvector_w,p,q,calc_error = True):
    
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
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
    plt.show()
def visualizeResultsBspline(results,p,q,knotvector_x, knotvector_y,surfacepoints=None):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            analitical.append(0.5*(xx**2-1)*(yy**2-1))
            d = mesh.distanceFromContur(xx,yy)
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum + (1-d)*0
            result.append(sum)
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,analitical)

    if surfacepoints:
        xPoints = []
        yPoints = []
        zPoints = []
        for i in surfacepoints:
            for q in i:
                xPoints.append(q[0])  
                yPoints.append(q[1])  
                zPoints.append(q[2]) 
        ax.scatter(xPoints,yPoints,zPoints)
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    plt.show()
def plotDistanceField():
    xPoints = []
    yPoints = []
    result = []

    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)
    for xx in x:
        for yy in y:
            xPoints.append(xx)
            yPoints.append(yy)
            result.append(sigmoid(mesh.distanceFromContur(xx,yy)))
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    plt.show()
def calculateError(surface, ctrlpts, result,k,l,weigths,knotvector_u,knotvector_w,p,q):
    zPoints = []
    zRes = []
    #result.append([0,0,0,0])
    #plotBsplineBasis(np.linspace(0,1,100), knotvector_u,p)
    for surfpoint in surface:
        for koordinate in surfpoint:
            res  = 0
            for i in range(0,k):
                for j in range(0,l):
                    res += result[k*i+j]*R2(k,l,i,j,koordinate[0],koordinate[1],weigths,knotvector_u,knotvector_w,p,q)
            zPoints.append(res) 
            zRes.append( 0.5*(koordinate[0]**2-1)*(koordinate[1]**2-1))
    MSE = (np.square(np.array(zRes)-np.array(zPoints))).mean()
    #print(f"MSE: {MSE}")
    return(MSE)
def calculateErrorBspline(surface, ctrlpts, results,k,l,weigths,knotvector_u,knotvector_w,p,q):
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
            analitical.append(0.5*(xx**2-1)*(yy**2-1))
            for xbasis in range(len(knotvector_u)-p-1):
                for ybasis in range(len(knotvector_w)-q-1):
                    sum += B(xx,p,xbasis,knotvector_u)*B(yy,q,ybasis,knotvector_w)*results[(len(knotvector_u)-p-1)*xbasis+ybasis]
            result.append(sum)
    
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    return MSE


#* TEST
if __name__ == "__main__":
    print("2D - Immersed - FEM.py")
    plotDistanceField()