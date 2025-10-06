from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh
import math
import torch
from bspline import Bspline

FUNCTION_CASE = 2
LARGER_DOMAIN = FUNCTION_CASE <=4 # if True, the domain is [-1,1]x[-1,1], otherwise [0,1]x[0,1]
print(f"Larger domain: {LARGER_DOMAIN}")
MAX_SUBDIVISION = 4
assert FUNCTION_CASE != 1
# Pre-compute Gauss quadrature data to avoid repeated calculations
_GAUSS_CACHE = {}

def _get_gauss_points(p):
    """Cache Gauss quadrature points and weights"""
    if p in _GAUSS_CACHE:
        return _GAUSS_CACHE[p]
    
    if p <= 2:
        g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
        w = np.array([1,1])
        gaussP_x = np.array([g[0],g[0],g[1],g[1]])
        gaussP_y = np.array([g[0],g[1],g[0],g[1]])
        gauss_weights = np.array([w[0],w[0],w[1],w[1]])
        num_gauss_points = 4
    else:
        g = np.array([-math.sqrt(3/5), 0, math.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
        gaussP_x = np.array([g[0],g[0],g[0],g[1],g[1],g[1],g[2],g[2],g[2]])
        gaussP_y = np.array([g[0],g[1],g[2],g[0],g[1],g[2],g[0],g[1],g[2]])
        gauss_weights = np.array([w[0]*w[0],w[1]*w[0],w[2]*w[0],w[0]*w[1],w[1]*w[1],w[2]*w[1],w[0]*w[2],w[1]*w[2],w[2]*w[2]])
        num_gauss_points = 9
    
    result = (gaussP_x, gaussP_y, gauss_weights, num_gauss_points)
    _GAUSS_CACHE[p] = result
    return result

# Vectorized function evaluations
def load_function_vectorized(x, y):
    """Vectorized version of load_function"""
    if FUNCTION_CASE == 1:
        return -8*x
    elif FUNCTION_CASE == 2:
        arg = (x**2 + y**2)*math.pi/2
        return -(-2*math.pi*np.sin(arg)-np.cos(arg)*(x**2 + y**2)*math.pi**2)
    elif FUNCTION_CASE == 3:
        return -8*x
    elif FUNCTION_CASE == 4:
        return -8*x
    elif FUNCTION_CASE == 5:#L-shape
        return 8*math.pi*math.pi*np.sin(2*math.pi*x)*np.sin(2*math.pi*y)
    elif FUNCTION_CASE == 6: #tube
        return -(x**2 + y**2)
    else:
        raise NotImplementedError

def dirichletBoundary_vectorized(x, y):
    """Vectorized version of dirichletBoundary"""
    if FUNCTION_CASE == 1:
        return np.full_like(x, 2)
    if FUNCTION_CASE == 2:
        return np.zeros_like(x)
    if FUNCTION_CASE == 3:
        return np.full_like(x, 2)
    if FUNCTION_CASE == 4:
        return x + 2*y
    elif FUNCTION_CASE == 5:
        return np.zeros_like(x)
    else: 
        raise NotImplementedError

def dirichletBoundaryDerivativeX_vectorized(x, y):
    """Vectorized version of dirichletBoundaryDerivativeX"""
    if FUNCTION_CASE <= 3:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 4:
        return np.ones_like(x)
    elif FUNCTION_CASE == 5:#L-shape
        return np.zeros_like(x)
    else: 
        raise NotImplementedError

def dirichletBoundaryDerivativeY_vectorized(x, y):
    """Vectorized version of dirichletBoundaryDerivativeY"""
    if FUNCTION_CASE <= 3:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 4:
        return np.full_like(x, 2)
    elif FUNCTION_CASE ==5:  #L-shape
        return np.zeros_like(x)
    else: 
        raise NotImplementedError

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
    elif FUNCTION_CASE == 5:#L-shape
        return 8*math.pi*math.pi*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
    elif FUNCTION_CASE == 6: #tube
        return -(x**2 + y**2)
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
    elif FUNCTION_CASE == 5: #L-shape
        return math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
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
    elif FUNCTION_CASE == 5:
        return 0
    else: raise NotImplementedError
def dirichletBoundaryDerivativeX(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 1
    elif FUNCTION_CASE == 5:#L-shape
        return 0
    else: raise NotImplementedError
def dirichletBoundaryDerivativeY(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 2
    elif FUNCTION_CASE ==5:  #L-shape
        return 0
    else: raise NotImplementedError

def element(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
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
def boundaryElement(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
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
    """Optimized subdivision function with reduced allocations and early termination"""
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    
    if level == MAXLEVEL:
        # At maximum level, perform Gauss quadrature on 4 subdomains
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        # Define the 4 subdomains
        subdomains = [
            (x1, halfx, y1, halfy),      # bottom-left
            (halfx, x2, y1, halfy),      # bottom-right  
            (x1, halfx, halfy, y2),      # top-left
            (halfx, x2, halfy, y2)       # top-right
        ]
        
        # Process all subdomains and accumulate results
        for sub_x1, sub_x2, sub_y1, sub_y2 in subdomains:
            Ks, Fs = GaussQuadrature(model, sub_x1, sub_x2, sub_y1, sub_y2, 
                                   i, j, p, q, knotvector_x, knotvector_y, Bspxi, Bspeta)
            K += Ks * 0.25  # More efficient than /4
            F += Fs * 0.25
    else:
        # Recursive subdivision
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        # Process 4 quadrants recursively
        quadrants = [
            (x1, halfx, y1, halfy),
            (x1, halfx, halfy, y2),
            (halfx, x2, y1, halfy),
            (halfx, x2, halfy, y2)
        ]
        
        for sub_x1, sub_x2, sub_y1, sub_y2 in quadrants:
            Kret, Fret = Subdivide(model, sub_x1, sub_x2, sub_y1, sub_y2,
                                 i, j, knotvector_x, knotvector_y, p, q, 
                                 Bspxi, Bspeta, level + 1, MAXLEVEL)
            K += Kret * 0.25
            F += Fret * 0.25
    
    return K, F

def GaussQuadrature(model,x1,x2,y1,y2,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta):
    # Use cached Gauss points
    gaussP_x, gaussP_y, gauss_weights, num_gauss_points = _get_gauss_points(p)
    
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    
    # Transform Gauss points to physical coordinates
    xi = (x2-x1)/2 * gaussP_x + (x2+x1)/2
    eta = (y2-y1)/2 * gaussP_y + (y2+y1)/2

    # Get distance function and derivatives
    d_,dx_,dy_ = mesh.distance_with_derivative_vect_trasformed(xi,eta,model)
    
    # Pre-compute valid gauss points (where d >= 0)
    valid_mask = d_ >= 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return K, F
    
    # Extract valid data only once
    d_valid = d_[valid_indices]
    dx_valid = dx_[valid_indices]
    dy_valid = dy_[valid_indices]
    weights_valid = gauss_weights[valid_indices]
    xi_valid = xi[valid_indices]
    eta_valid = eta[valid_indices]
    
    # Compute B-spline values only for valid points
    bxi_valid = Bspxi.collmat(xi_valid)
    beta_valid = Bspeta.collmat(eta_valid) 
    dbdxi_valid = Bspxi.collmat(xi_valid, 1)
    dbdeta_valid = Bspeta.collmat(eta_valid, 1)
    
    # Pre-compute function values for all valid points (vectorized)
    f_values = load_function_vectorized(xi_valid, eta_valid)
    dirichlet_values = dirichletBoundary_vectorized(xi_valid, eta_valid)
    dirichlet_dx = dirichletBoundaryDerivativeX_vectorized(xi_valid, eta_valid)
    dirichlet_dy = dirichletBoundaryDerivativeY_vectorized(xi_valid, eta_valid)
    
    # Create basis function index arrays
    x_range = np.arange(i-p, i+1)
    y_range = np.arange(j-q, j+1)
    
    # Pre-compute all local indices for matrix assembly
    local_indices = []
    for ii, xi_idx in enumerate(x_range):
        for jj, yi_idx in enumerate(y_range):
            local_i = ii
            local_j = jj
            F_idx = local_i * (q+1) + local_j
            local_indices.append((local_i, local_j, F_idx, xi_idx, yi_idx))
    
    # Main computation loop over valid Gauss points
    for gp_idx in range(len(valid_indices)):
        d = d_valid[gp_idx].item()
        dx = dx_valid[gp_idx].item() 
        dy = dy_valid[gp_idx].item()
        weight = weights_valid[gp_idx]
        
        bxi = bxi_valid[gp_idx]
        beta = beta_valid[gp_idx]
        dbdxi = dbdxi_valid[gp_idx]
        dbdeta = dbdeta_valid[gp_idx]
        
        # Pre-compute all basis function values for this Gauss point
        N_values = []
        dN_xi_values = []
        dN_eta_values = []
        corr_xi_values = []
        corr_eta_values = []
        
        for _, _, _, xi_idx, yi_idx in local_indices:
            # Handle both array and scalar cases for basis function evaluations
            if np.isscalar(beta) or beta.ndim == 0:
                beta_val = beta
            else:
                beta_val = beta[yi_idx] if yi_idx < len(beta) else 0.0
                
            if np.isscalar(bxi) or bxi.ndim == 0:
                bxi_val = bxi
            else:
                bxi_val = bxi[xi_idx] if xi_idx < len(bxi) else 0.0
                
            if np.isscalar(dbdxi) or dbdxi.ndim == 0:
                dbdxi_val = dbdxi
            else:
                dbdxi_val = dbdxi[xi_idx] if xi_idx < len(dbdxi) else 0.0
                
            if np.isscalar(dbdeta) or dbdeta.ndim == 0:
                dbdeta_val = dbdeta
            else:
                dbdeta_val = dbdeta[yi_idx] if yi_idx < len(dbdeta) else 0.0
            
            N = beta_val * bxi_val
            dN_xi = dbdxi_val * beta_val
            dN_eta = bxi_val * dbdeta_val
            
            corr_xi = dN_xi * d + dx * N
            corr_eta = dN_eta * d + dy * N
            
            N_values.append(N)
            dN_xi_values.append(dN_xi)
            dN_eta_values.append(dN_eta)
            corr_xi_values.append(corr_xi)
            corr_eta_values.append(corr_eta)
        
        # Vectorized K matrix assembly
        for idx_i, (local_i, local_j, F_idx_i, _, _) in enumerate(local_indices):
            corr_xi_i = corr_xi_values[idx_i]
            corr_eta_i = corr_eta_values[idx_i]
            
            for idx_j, (local_k, local_l, F_idx_j, _, _) in enumerate(local_indices):
                corr_xi_j = corr_xi_values[idx_j]
                corr_eta_j = corr_eta_values[idx_j]
                
                K_contrib = (corr_xi_i * corr_xi_j + corr_eta_i * corr_eta_j) * weight
                K_row = local_i * (q+1) + local_j
                K_col = local_k * (q+1) + local_l
                K[K_row, K_col] += K_contrib
            
            # F vector assembly
            N_i = N_values[idx_i]
            Ni_corr = d * N_i
            fi = f_values[gp_idx]
            dirichlet_val = dirichlet_values[gp_idx]
            ddir_x = dirichlet_dx[gp_idx] 
            ddir_y = dirichlet_dy[gp_idx]
            
            F_contrib = (fi * Ni_corr + 
                        (corr_xi_i * (dx * dirichlet_val + d * ddir_x) +
                         corr_eta_i * (dy * dirichlet_val + d * ddir_y)) - 
                        (ddir_x * corr_xi_i + ddir_y * corr_eta_i)) * weight
            
            F[F_idx_i] += F_contrib
    
    return K, F

# Cache for Bspline objects to avoid repeated creation
_BSPLINE_CACHE = {}

def _get_bspline_objects(knotvector_x, knotvector_y, p, q):
    """Cache Bspline objects to avoid repeated creation"""
    # Create hashable key from arrays
    key = (tuple(knotvector_x.tolist() if hasattr(knotvector_x, 'tolist') else knotvector_x), 
           tuple(knotvector_y.tolist() if hasattr(knotvector_y, 'tolist') else knotvector_y), 
           p, q)
    
    if key in _BSPLINE_CACHE:
        return _BSPLINE_CACHE[key]
    
    Bspxi = Bspline(knotvector_x, p)
    Bspeta = Bspline(knotvector_y, q)
    result = (Bspxi, Bspeta)
    _BSPLINE_CACHE[key] = result
    return result

def elementChoose(model,p,q,knotvector_x, knotvector_y,i,j,etype=None):
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    
    # Use pre-allocated tensor to avoid repeated allocations
    points = torch.tensor([[x1,y1],[x2,y1],[x1,y2],[x2,y2]], dtype=torch.float32)
    distances = model(points)
    
    # Optimized element classification using vectorized operations
    distances_np = distances.detach().numpy().flatten()
    innerElement = np.all(distances_np >= 0)
    outerElement = np.all(distances_np < 0)
    
    # Get cached Bspline objects
    if innerElement or not outerElement:  # Need Bspline objects
        Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
    
    if innerElement: #regular element
        if etype is not None: etype["inner"] +=1
        Ke, Fe = element(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta)
    elif outerElement:
        Ke = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
        Fe = np.zeros((p+1)*(q+1))
        if etype is not None: etype["outer"] +=1
    else:
        Ke, Fe = boundaryElement(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta)
        if etype is not None: etype["boundary"] +=1
    
    if etype is not None: return Ke, Fe, etype
    return Ke, Fe
def elementTypeChoose(knotvector_x, knotvector_y,i,j,etype=None):
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    """distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]"""
    points = torch.tensor(np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]]),dtype=torch.float32)
    distances = model(points)
    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point<0:
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

def visualizeResultsBspline(model,results,p,q,knotvector_x, knotvector_y,surfacepoints=None,larger_domain=True):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    if larger_domain:
        x = np.linspace(-1,1,40)
        y = np.linspace(-1,1,40)
    else:
        x = np.linspace(0,1,40)
        y = np.linspace(0,1,40)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model)
            analitical.append(solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            try:
                result.append(sum.item())
            except:
                result.append(sum)
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,analitical)

    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    plt.show()
def calculateErrorBspline(model,results,p,q,knotvector_x, knotvector_y,larger_domain=True):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    if larger_domain:
        x = np.linspace(-1,1,40)
        y = np.linspace(-1,1,40)
    else:
        x = np.linspace(0,1,40)
        y = np.linspace(0,1,40)
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
def plotErrorHeatmap(model,results,knotvector_x,knotvector_y,p,q,larger_domain = True,N=150):

    marg = 0.1
    if larger_domain:
        x = np.linspace(-1-marg,1+marg,N)
        y = np.linspace(-1-marg,1+marg,N)
    else:
        x = np.linspace(0-marg,1+marg,N)
        y = np.linspace(0-marg,1+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_N=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            Z_N[idxx,idxy] = sum
    #MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    #print(f"MSE: {MSE}")
    #return MSE
    plt.contourf(X, Y, Z_N,levels=20)
    points = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0),(1.0,0.5),(0.5,0.5),(0.5,1.0),(0.0, 1.0)]

    # Plot red lines between the points
    for i in range(len(points)-1):
            plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'r-')

    #plt.axis('equal')
    #highlight_level = 0.0
    #plt.contour(X, Y, Z, levels=[highlight_level], colors='red')
    plt.colorbar()
    plt.title('Solution of the PDE')
    plt.grid(True)
    plt.show()
def plotResultHeatmap(model,results,knotvector_x,knotvector_y,p,q,larger_domain = True,N=150):
    marg = 0.05
    if larger_domain:
        x = np.linspace(-1-marg,1+marg,N)
        y = np.linspace(-1-marg,1+marg,N)
    else:
        x = np.linspace(0-marg,1+marg,N)
        y = np.linspace(0-marg,1+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_A=np.zeros((N,N))
    Z_N=np.zeros((N,N))
    ERR=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model)
            Z_A[idxx,idxy] = solution_function(xx,yy) if d>=0 else 0
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            Z_N[idxx,idxy] = sum
    #MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    #print(f"MSE: {MSE}")
    #return MSE
    ERR = np.abs(Z_N-Z_A)
    plt.contourf(X, Y, Z_A,levels=20)
    #plt.axis('equal')
    plt.colorbar()
    if larger_domain:
        highlight_level = 0.0
        plt.contour(X, Y, Z_A, levels=[highlight_level], colors='red') 
    else:
        points = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0),(1.0,0.5),(0.5,0.5),(0.5,1.0),(0.0, 1.0)]

        # Plot red lines between the points
        for i in range(len(points)-1):
                plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'r-')

    # Show the plot
    
    plt.title('Solution')
    plt.grid(True)
    plt.show()
#* TEST
if __name__ == "__main__":
    from NeuralImplicit import Siren
    siren_model_kor_jo = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
    siren_model_kor_jo.load_state_dict(torch.load('siren_model_kor_jo.pth',weights_only=True,map_location=torch.device('cpu')))
    siren_model_kor_jo.eval()
    model = siren_model_kor_jo
    test_values = [20,30,40,50,60,80,120]
    esize = [1/(nd+1) for nd in test_values]
    orders = [2]
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
                    etype = elementTypeChoose(knotvector_u,knotvector_w,elemx,elemy,etype)
            etypes.append(etype)
    print(etypes)
