from NURBS import *
from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh
import math
import torch
import numpy as np
FUNCTION_CASE = 2
MAX_SUBDIVISION = 4
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

def elemantBspline(model,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts,Bspxi,Bspeta):
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
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=0)
    return K,F
def boundaryElementBspline(model,r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly,Bspxi,Bspeta):
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
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=MAX_SUBDIVISION)
    return K,F
def Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    halfy = (y1+y2)/2
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    r=1 #! Ez hardcodolva van!!!
    if level == MAXLEVEL:
        #first
        Ks,Fs = GaussQuadrature(model,x1, halfx,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #second
        Ks,Fs = GaussQuadrature(model,halfx, x2,y1,halfy,r,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #third
        Ks,Fs = GaussQuadrature(model,x1, halfx,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #fourth
        Ks,Fs = GaussQuadrature(model,halfx, x2,halfy,y2,r,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        return K,F
    else:
        Kret,Fret=Subdivide(model,x1,halfx,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,x1,halfx,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        return K,F
def GaussQuadrature_old(model,x1,x2,y1,y2,r,i,j,p,q,knotvector_x,knotvector_y):
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
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += w[idxx]*w[idxy]*(((diCorrXi)*(djCorrXi) + (diCorrEta)*(djCorrEta)))
                    fi = load_function(xi, eta)
                    Ni_corr = d*Ni
                    dirichlet_xi_eta = dirichletBoundary(xi,eta)
                    Ddirichlet_X = dirichletBoundaryDerivativeX(xi,eta)
                    Ddirichlet_Y = dirichletBoundaryDerivativeY(xi,eta)
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += w[idxx]*w[idxy]*(fi*Ni_corr + (diCorrXi*(dx*dirichlet_xi_eta+d*Ddirichlet_X) + diCorrEta*(dy*dirichlet_xi_eta+d*Ddirichlet_Y)) - (Ddirichlet_X*diCorrXi + Ddirichlet_Y*diCorrEta))
    return K,F
from bspline import Bspline
import numpy as np
import math

def GaussQuadrature_gpt(model, x1, x2, y1, y2, r, i, j, p, q, knotvector_x, knotvector_y, Bspxi, Bspeta):
    g = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
    w = np.array([1, 1])

    # Initialize matrices and precompute constants
    dof = (p + 1) * (q + 1)
    K = np.zeros((dof, dof))
    F = np.zeros(dof)

    # Precompute Gaussian points and values
    gaussP_x, gaussP_y = np.meshgrid(g, g)
    xi = ((x2 - x1) / 2) * gaussP_x.flatten() + ((x2 + x1) / 2)
    eta = ((y2 - y1) / 2) * gaussP_y.flatten() + ((y2 + y1) / 2)

    # Precompute spline basis and derivatives for xi and eta
    bxi_ = Bspxi.collmat(xi)
    beta_ = Bspeta.collmat(eta)
    dbdxi_ = Bspxi.collmat(xi, 1)
    dbdeta_ = Bspeta.collmat(eta, 1)

    # Distance and derivative values for each Gauss point
    d_, dx_, dy_ = mesh.distance_with_derivative_vect(xi, eta, model)
    
    # Precompute boundary and load function values for all Gauss points
    dirichlet_vals = np.array([dirichletBoundary(x, y) for x, y in zip(xi, eta)])
    dirichlet_dX = np.array([dirichletBoundaryDerivativeX(x, y) for x, y in zip(xi, eta)])
    dirichlet_dY = np.array([dirichletBoundaryDerivativeY(x, y) for x, y in zip(xi, eta)])
    fi_vals = np.array([load_function(x, y) for x, y in zip(xi, eta)])

    # Vectorized iteration over Gauss points
    for idx in range(4):
        d = d_[idx]
        if d < 0:
            continue  # Skip negative distances

        dx, dy = dx_[idx], dy_[idx]
        bxi, beta = bxi_[idx], beta_[idx]
        dbdxi, dbdeta = dbdxi_[idx], dbdeta_[idx]

        # Calculate corrections for basis function pairs
        for xbasisi in range(i - p, i + 1):
            for ybasisi in range(j - q, j + 1):
                Ni = beta[ybasisi] * bxi[xbasisi]
                dNidxi = dbdxi[xbasisi] * beta[ybasisi]
                dNidEta = bxi[xbasisi] * dbdeta[ybasisi]

                # Corrections
                diCorrXi = dNidxi * d + dx * Ni
                diCorrEta = dNidEta * d + dy * Ni
                K_idx_i = (p + 1) * (xbasisi - (i - p)) + (ybasisi - (j - q))

                # Update F vector in a fully vectorized way
                F[K_idx_i] += (
                    fi_vals[idx] * d * Ni
                    + diCorrXi * (dx * dirichlet_vals[idx] + d * dirichlet_dX[idx])
                    + diCorrEta * (dy * dirichlet_vals[idx] + d * dirichlet_dY[idx])
                    - dirichlet_dX[idx] * diCorrXi
                    - dirichlet_dY[idx] * diCorrEta
                )

                # Vectorized update of K matrix
                for xbasisj in range(i - p, i + 1):
                    for ybasisj in range(j - q, j + 1):
                        dNjdxi = dbdxi[xbasisj] * beta[ybasisj]
                        dNjdeta = bxi[xbasisj] * dbdeta[ybasisj]
                        Nj = beta[ybasisj] * bxi[xbasisj]

                        djCorrXi = dNjdxi * d + dx * Nj
                        djCorrEta = dNjdeta * d + dy * Nj

                        # Direct indexing into K
                        K[K_idx_i, (p + 1) * (xbasisj - (i - p)) + (ybasisj - (j - q))] += (
                            diCorrXi * djCorrXi + diCorrEta * djCorrEta
                        )
    
    return K, F

def GaussQuadrature(model,x1,x2,y1,y2,r,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta):
    
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    Bspxi = Bspline(knotvector_x,p)
    Bspeta = Bspline(knotvector_y,q)
    gaussP_x = np.array([g[0],g[0],g[1],g[1]])
    gaussP_y = np.array([g[0],g[1],g[0],g[1]])
    xi = (x2-x1)/2 * gaussP_x + (x2+x1)/2
    eta = (y2-y1)/2 * gaussP_y + (y2+y1)/2

    d_,dx_,dy_ = mesh.distance_with_derivative_vect(xi,eta,model)
    bxi_ = Bspxi.collmat(xi)
    beta_ = Bspeta.collmat(eta)
    dbdxi_ = Bspxi.collmat(xi,1)
    dbdeta_ = Bspeta.collmat(eta,1)
    for idx in range(4): #iterate throug Gauss points functions in x 
            #d = mesh.distanceFromContur(xi,eta,model)
            d = d_[idx].item()
            if d<0: continue
            dx = dx_[idx].item()
            dy = dy_[idx].item()
            bxi = bxi_[idx]
            beta = beta_[idx]
            dbdxi = dbdxi_[idx]
            dbdeta = dbdeta_[idx]
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    dNidxi = dbdxi[xbasisi]*beta[ybasisi]
                    dNidEta = bxi[xbasisi]*dbdeta[ybasisi]
                    Ni = beta[ybasisi]*bxi[xbasisi]
                    diCorrXi = dNidxi*d + dx * Ni
                    diCorrEta = dNidEta*d + dy * Ni
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dbdxi[xbasisj]*beta[ybasisj]
                            dNjdeta = bxi[xbasisj]*dbdeta[ybasisj]
                            Nj = beta[ybasisj]*bxi[xbasisj]
                            #correction with the distance function
                            djCorrXi = dNjdxi*d + dx * Nj
                            djCorrEta = dNjdeta*d + dy * Nj
                            #! In the line below the weigths have been taken out
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += (((diCorrXi)*(djCorrXi) + (diCorrEta)*(djCorrEta)))
                    fi = load_function(xi[idx], eta[idx])
                    Ni_corr = d*Ni
                    dirichlet_xi_eta = dirichletBoundary(xi[idx],eta[idx])
                    Ddirichlet_X = dirichletBoundaryDerivativeX(xi[idx],eta[idx])
                    Ddirichlet_Y = dirichletBoundaryDerivativeY(xi[idx],eta[idx])
                    #! In the line below the weigths have been taken out
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += (fi*Ni_corr + (diCorrXi*(dx*dirichlet_xi_eta+d*Ddirichlet_X) + diCorrEta*(dy*dirichlet_xi_eta+d*Ddirichlet_Y)) - (Ddirichlet_X*diCorrXi + Ddirichlet_Y*diCorrEta))
    return K,F
def GaussQuadrature_opt(model, x1, x2, y1, y2, r, i, j, p, q, knotvector_x, knotvector_y):
    g = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])  # Gauss points
    w = np.array([1, 1])  # Weights
    K = np.zeros(((p + 1) * (q + 1), (p + 1) * (q + 1)))
    F = np.zeros((p + 1) * (q + 1))

    # Calculate xi and eta for all combinations of Gauss points
    gpx = (x2 - x1) / 2 * g[:, None] + (x2 + x1) / 2  # Shape: (2, 1)
    gpy = (y2 - y1) / 2 * g + (y2 + y1) / 2  # Shape: (2,)

    # Prepare for distance calculations
    points = np.array(np.meshgrid(gpx.flatten(), gpy.flatten())).T.reshape(-1, 2)
    
    # Calculate distances and derivatives
    d, dx, dy = mesh.distance_with_derivative(points[:, 0].T, points[:, 1].T, model)

    # Reshape d, dx, dy to 2x2 for processing
    d = d.reshape(2, 2)
    dx = dx.reshape(2, 2)
    dy = dy.reshape(2, 2)

    # Mask for valid distances
    valid_mask = d >= 0

    # Calculate basis functions and their derivatives for valid points
    xbasis_indices = np.arange(i - p, i + 1)
    ybasis_indices = np.arange(j - q, j + 1)

    # Vectorize B-spline calculations
    N_x = np.array([[B(gpy[jj], q, ybasisi, knotvector_y) for jj in range(2)] for ybasisi in ybasis_indices])
    N_y = np.array([[B(gpx[ii], p, xbasisi, knotvector_x) for ii in range(2)] for xbasisi in xbasis_indices])

    for idxx in range(2):  # Loop through Gauss points in x direction
        for idxy in range(2):  # Loop through Gauss points in y direction
            if not valid_mask[idxx, idxy]:
                continue  # Skip invalid points

            xi, eta = gpx[idxx, 0], gpy[idxy]

            # dNdxi and dNidEta
            test = dBdXi(xi, p, xbasis_indices[:, None], knotvector_x)
            dNidxi =  test* N_y[idxy]
            dNidEta = N_x[idxx] * dBdXi(eta, q, ybasis_indices, knotvector_y)

            Ni = N_x[idxx] * N_y[idxy]

            # Calculate corrections
            diCorrXi = dNidxi * d[idxx, idxy] + dx[idxx, idxy] * Ni
            diCorrEta = dNidEta * d[idxx, idxy] + dy[idxx, idxy] * Ni

            for xbasisj in range(i - p, i + 1):
                for ybasisj in range(j - q, j + 1):
                    Nj = N_x[idxy] * N_y[idxx]  # Vectorized basis function
                    dNjdxi = dBdXi(xi, p, xbasisj, knotvector_x) * N_y[ybasisj]
                    dNjdeta = N_x[ybasisj] * dBdXi(eta, q, ybasisj, knotvector_y)

                    # Correction for the distance function
                    djCorrXi = dNjdxi * d[idxx, idxy] + dx[idxx, idxy] * Nj
                    djCorrEta = dNjdeta * d[idxx, idxy] + dy[idxx, idxy] * Nj

                    K[(p + 1) * (xbasis_indices[0] - (i - p)) + (ybasis_indices[0] - (j - q))][
                        (p + 1) * (xbasis_indices[0] - (i - p)) + (ybasis_indices[0] - (j - q))] += (
                            w[idxx] * w[idxy] * (
                                (diCorrXi * djCorrXi + diCorrEta * djCorrEta))
                        )

            fi = load_function(xi, eta)
            Ni_corr = d[idxx, idxy] * Ni
            dirichlet_xi_eta = dirichletBoundary(xi, eta)
            Ddirichlet_X = dirichletBoundaryDerivativeX(xi, eta)
            Ddirichlet_Y = dirichletBoundaryDerivativeY(xi, eta)

            # Update F
            F[(p + 1) * (xbasis_indices[0] - i + p) + (ybasis_indices[0] - j + q)] += (
                w[idxx] * w[idxy] * (
                    fi * Ni_corr  +
                    (diCorrXi * (dx * dirichlet_xi_eta + d[idxx, idxy] * Ddirichlet_X) +
                     diCorrEta * (dy * dirichlet_xi_eta + d[idxx, idxy] * Ddirichlet_Y)) -
                    (Ddirichlet_X * diCorrXi + Ddirichlet_Y * diCorrEta)
                )
            )

    return K, F
def elementChoose(model,Nurbs_fun,r,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts,etype=None):
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
        Bspxi = Bspline(knotvector_x,p)
        Bspeta = Bspline(knotvector_y,q)
        #Ke, Fe = elemantBspline(p,q,knotvector_x, knotvector_y, None,i,j,nControlx, nControly,weigths,ctrlpts)
        if etype is not None: etype["inner"] +=1
        Ke, Fe = elemantBspline(model,p,q,knotvector_x, knotvector_y, ed,i,j,nControlx, nControly,weigths,ctrlpts,Bspxi,Bspeta)
    elif outerElement:
        Ke = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
        Fe = np.zeros((p+1)*(q+1))
        if etype is not None: etype["outer"] +=1
    else:
        Bspxi = Bspline(knotvector_x,p)
        Bspeta = Bspline(knotvector_y,q)
        Ke, Fe = boundaryElementBspline(model,r,p,q,knotvector_x, knotvector_y,i,j,nControlx, nControly,Bspxi,Bspeta)
        if etype is not None: etype["boundary"] +=1
    if etype is not None: return Ke, Fe, etype
    return Ke, Fe
def elementTypeChoose(r,knotvector_x, knotvector_y,i,j,etype=None):
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
        if etype is not None: etype["inner"] +=1
    elif outerElement:
        if etype is not None: etype["outer"] +=1
    else:
        if etype is not None: etype["boundary"] +=1
    if etype is not None: return etype
    return None
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
    #inv = np.linalg.inv(K_reduced)
    u_reduced = np.linalg.solve(K_reduced,F_reduced)
    #pinv = np.dot(np.linalg.inv(np.dot(np.transpose(K_reduced),K_reduced)),K_reduced)
    #svd_inv = svd_inverse(K_reduced)
    #reg_inv = regularized_inverse(K_reduced)
    #u_reduced = np.dot(inv,F_reduced)
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
    test_values = [20,30,40]
    esize = [1/(nd+1) for nd in test_values]
    orders = [3]
    fig,ax = plt.subplots()
    for order in orders:
        accuracy = []
        etypes = []
        for division in test_values:
            etype = {"outer":0,"inner":0,"boundary":0}
            default = mesh.getDefaultValues(div=division,order=order,delta=0.005)
            x0, y0,x1,y1,xDivision,yDivision,p,q = default
            knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
            assert p==q and xDivision == yDivision
            for elemx in range(p,p+xDivision+1):
                for elemy in range(q,q+xDivision+1):
                    etype = elementTypeChoose(1,knotvector_u,knotvector_w,elemx,elemy,etype)
            etypes.append(etype)
    print(etypes)
