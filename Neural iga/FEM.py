from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh
import math
FUNCTION_CASE = 2
MAX_SUBDIVISION = 3
def load_function(x,y):
    #! -f(x)
    if FUNCTION_CASE == 1:
        return -8*x
    elif FUNCTION_CASE == 2:
        arg = (x**2 + y**2)*math.pi/2
        return -(-2*math.pi*math.sin(arg)-math.cos(arg)*(x**2 + y**2)*math.pi**2)
    elif FUNCTION_CASE == 3:
        return -8*x
    elif FUNCTION_CASE == 4:
        return -8*x
    else:
        raise NotImplementedError
def solution_function(x,y):
    if FUNCTION_CASE == 1:
        return x*(x**2 + y**2 -1)
    elif FUNCTION_CASE == 2:
        return math.cos((x**2 + y**2)*math.pi/2)
    elif FUNCTION_CASE == 3:
        return x*(x**2 + y**2 -1) + 2
    elif FUNCTION_CASE == 4:
        return x*(x**2 + y**2 -1) + x +2*y
    else: raise NotImplementedError
def dirichletBoundary(x,y):
    if FUNCTION_CASE == 1:
        return 2
    if FUNCTION_CASE == 2:
        return 0
    if FUNCTION_CASE == 3:
        return 2
    if FUNCTION_CASE == 4:
        return x+2*y
    else: raise NotImplementedError
def dirichletBoundaryDerivativeX(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 1
    else: raise NotImplementedError
def dirichletBoundaryDerivativeY(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 2
    else: raise NotImplementedError

def elemantBspline(model,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
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
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,level=0,MAXLEVEL=0)
    return K,F
def boundaryElementBspline(model,r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly):
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
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,level=0,MAXLEVEL=MAX_SUBDIVISION)
    return K,F
def Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    halfy = (y1+y2)/2
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    r=1 #! Ez hardcodolva van!!!
    if level == MAXLEVEL:
        #first
        Ks,Fs = GaussQuadrature(model,x1, halfx,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks/4
        F+=Fs/4
        #second
        Ks,Fs = GaussQuadrature(model,halfx, x2,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks/4
        F+=Fs/4
        #third
        Ks,Fs = GaussQuadrature(model,x1, halfx,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks/4
        F+=Fs/4
        #fourth
        Ks,Fs = GaussQuadrature(model,halfx, x2,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y)
        K+=Ks/4
        F+=Fs/4
        return K,F
    else:
        Kret,Fret=Subdivide(model,x1,halfx,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,x1,halfx,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        return K,F
def GaussQuadrature(model,x1,x2,y1,y2,r,i,j,p,q,knotvector_x,knotvector_y):
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        for idxy,gpy in enumerate(g): #iterate throug Gauss points functions in y direction
            xi = (x2-x1)/2 * gpx + (x2+x1)/2
            eta = (y2-y1)/2 * gpy + (y2+y1)/2
            #d = mesh.distanceFromContur(xi,eta,model)
            d,dx,dy = mesh.distance_with_derivative(xi,eta,model)
            if d<0: continue
            Jxi = (x2-x1)/2
            Jeta = (y2-y1)/2
            Jacobi = Jxi*Jeta
            Jacobi = 1
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    dNidxi = dBdXi(xi,p,xbasisi,knotvector_x)*B(eta,q,ybasisi,knotvector_y)
                    dNidEta = B(xi, p, xbasisi,knotvector_x)*dBdXi(eta,q,ybasisi,knotvector_y)
                    Ni = B(eta,q,ybasisi,knotvector_y)*B(xi,p,xbasisi,knotvector_x)
                    diCorrXi = dNidxi*d + dx * Ni
                    diCorrEta = dNidEta*d + dy * Ni
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dBdXi(xi,p,xbasisj,knotvector_x)*B(eta,q,ybasisj,knotvector_y)
                            dNjdeta = B(xi,p,xbasisj,knotvector_x)*dBdXi(eta,q,ybasisj,knotvector_y)
                            Nj = B(eta,q,ybasisj,knotvector_y)*B(xi,p,xbasisj,knotvector_x)
                            #correction with the distance function
                            djCorrXi = dNjdxi*d + dx * Nj
                            djCorrEta = dNjdeta*d + dy * Nj
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((diCorrXi)*(djCorrXi) + (diCorrEta)*(djCorrEta))*Jacobi)
                    fi = load_function(xi, eta)
                    Ni_corr = d*Ni
                    dirichlet_xi_eta = dirichletBoundary(xi,eta)
                    Ddirichlet_X = dirichletBoundaryDerivativeX(xi,eta)
                    Ddirichlet_Y = dirichletBoundaryDerivativeY(xi,eta)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni_corr*Jacobi + (diCorrXi*(dx*dirichlet_xi_eta+d*Ddirichlet_X) + diCorrEta*(dy*dirichlet_xi_eta+d*Ddirichlet_Y)) - (Ddirichlet_X*diCorrXi + Ddirichlet_Y*diCorrEta))
    return K,F
def elementChoose(model,Nurbs_fun,r,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts):
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
        if point>r:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        #Ke, Fe = elemantBspline(p,q,knotvector_x, knotvector_y, None,i,j,nControlx, nControly,weigths,ctrlpts)
        Ke, Fe = elemantBspline(model,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts)
    elif outerElement:
        Ke = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
        Fe = np.zeros((p+1)*(q+1))
    else:
        Ke, Fe = boundaryElementBspline(model,r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly)
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

def solveWeak(K,F):
    zero_rows = np.all(K == 0, axis=1)
    zero_cols = np.all(K == 0, axis=0)
    zero_f = zero_rows
    # Remove zero rows and columns
    K_reduced = K[~zero_rows][:, ~zero_cols]
    F_reduced = F[~zero_f]
    u = np.zeros(len(F))
    #u_reduced = np.dot(np.linalg.inv(K_reduced),F_reduced)
    inv = np.linalg.inv(K_reduced)
    #pinv = np.dot(np.linalg.inv(np.dot(np.transpose(K_reduced),K_reduced)),K_reduced)
    #svd_inv = svd_inverse(K_reduced)
    #reg_inv = regularized_inverse(K_reduced)
    u_reduced = np.dot(inv,F_reduced)
    u[~zero_f] = u_reduced
    return u

def visualizeResultsBspline(model,results,p,q,knotvector_x, knotvector_y,surfacepoints=None):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(-1,1,20)
    y = np.linspace(-1,1,20)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            analitical.append(solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            result.append(sum)
    #fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    #ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    #ax.scatter(xPoints,yPoints,analitical)

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
            result.append(mesh.distanceFromContur(xx,yy))
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    plt.show()
def calculateErrorBspline(model,results,p,q,knotvector_x, knotvector_y):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(-1,1,20)
    y = np.linspace(-1,1,20)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            analitical.append(solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            result.append(sum)
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    return MSE

#* TEST
if __name__ == "__main__":
    print("2D - Immersed - FEM.py")
    plotDistanceField()