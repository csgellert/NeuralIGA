from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh
import math
import torch
from bspline import Bspline
from tqdm import tqdm

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)
NP_DTYPE = np.float64
TORCH_DTYPE = torch.float64

FUNCTION_CASE = 1
MAX_SUBDIVISION = 4
#assert FUNCTION_CASE != 1
# Pre-compute Gauss quadrature data to avoid repeated calculations
_GAUSS_CACHE = {}
DOMAIN = {"x1": -1, "x2": 1, "y1": -1, "y2": 1}

def _get_gauss_points(p):
    """Cache Gauss quadrature points and weights.
    
    For B-spline of order p, the integrand in the stiffness matrix involves
    products of derivatives of order p-1, so we need to integrate polynomials
    of degree 2*(p-1). n Gauss points can exactly integrate polynomials of 
    degree 2n-1, so we need n >= p Gauss points per direction.
    
    Additionally, for higher accuracy with the distance function (which may be
    non-polynomial), we use one extra point.
    """
    if p in _GAUSS_CACHE:
        return _GAUSS_CACHE[p]
    
    if p <= 2:
        # 2x2 Gauss quadrature - integrates exactly up to degree 3
        g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif p == 3:
        # 3x3 Gauss quadrature - integrates exactly up to degree 5
        g = np.array([-math.sqrt(3/5), 0.0, math.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
    elif p == 4:
        # 4x4 Gauss quadrature - integrates exactly up to degree 7
        g1 = math.sqrt(3/7 - 2/7*math.sqrt(6/5))
        g2 = math.sqrt(3/7 + 2/7*math.sqrt(6/5))
        w1 = (18 + math.sqrt(30)) / 36
        w2 = (18 - math.sqrt(30)) / 36
        g = np.array([-g2, -g1, g1, g2])
        w = np.array([w2, w1, w1, w2])
    else:
        # 5x5 Gauss quadrature for p >= 5 - integrates exactly up to degree 9
        g1 = 1/3 * math.sqrt(5 - 2*math.sqrt(10/7))
        g2 = 1/3 * math.sqrt(5 + 2*math.sqrt(10/7))
        w1 = (322 + 13*math.sqrt(70)) / 900
        w2 = (322 - 13*math.sqrt(70)) / 900
        w0 = 128/225
        g = np.array([-g2, -g1, 0.0, g1, g2])
        w = np.array([w2, w1, w0, w1, w2])
    
    # Build 2D tensor product quadrature
    n_1d = len(g)
    gaussP_x = np.repeat(g, n_1d)  # [g0,g0,...,g0, g1,g1,...,g1, ...]
    gaussP_y = np.tile(g, n_1d)    # [g0,g1,...,gn, g0,g1,...,gn, ...]
    gauss_weights = np.outer(w, w).flatten()  # w_i * w_j
    num_gauss_points = n_1d * n_1d
    
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
    elif FUNCTION_CASE == 7: #double circle
        arg = (x**2 + y**2)*math.pi
        return 4*math.pi*np.cos(arg) -4*math.pi*arg*np.sin(arg)
    else:
        raise NotImplementedError

def dirichletBoundary_vectorized(x, y):
    """Vectorized version of dirichletBoundary"""
    if FUNCTION_CASE == 1:
        return np.full_like(x, 0)
    if FUNCTION_CASE == 2:
        return np.zeros_like(x)
    if FUNCTION_CASE == 3:
        return np.full_like(x, 2)
    if FUNCTION_CASE == 4:
        return x + 2*y
    elif FUNCTION_CASE == 5:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 7:
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
    elif FUNCTION_CASE == 7:
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
    elif FUNCTION_CASE == 7:
        return np.zeros_like(x)
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
        return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    elif FUNCTION_CASE == 7: #double circle
        return math.sin(math.pi*(x**2 + y**2))
    else: raise NotImplementedError
def solution_function_derivative_x(x,y):
    if FUNCTION_CASE == 1:
        return 3*x**2 + y**2 -1
    elif FUNCTION_CASE == 2:
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi/2
        return -math.pi*x*math.sin(arg)
    elif FUNCTION_CASE == 3:
        return 3*x**2 + y**2 -1
    elif FUNCTION_CASE == 4:
        raise NotImplementedError
        return 3*x**2 + y**2 -1 +1
    elif FUNCTION_CASE == 5: #L-shape
        return 2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    elif FUNCTION_CASE ==7: #double circle
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi
        return 2*math.pi*x*math.cos(arg)
    else: raise NotImplementedError
def solution_function_derivative_y(x,y):
    if FUNCTION_CASE == 1:
        return 2*x*y
    elif FUNCTION_CASE == 2:
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi/2
        return -math.pi*y*math.sin(arg)
    elif FUNCTION_CASE == 3:
        return 2*x*y
    elif FUNCTION_CASE == 4:
        raise NotImplementedError
        return 2*x*y +2
    elif FUNCTION_CASE == 5: #L-shape
        return 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    elif FUNCTION_CASE ==7: #double circle
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi
        return 2*math.pi*y*math.cos(arg)
    else: raise NotImplementedError

def element(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
    """Process inner element with minimal subdivision"""
    assert q==p
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    return Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=0)
def boundaryElement(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
    """Process boundary element with maximum subdivision for accuracy"""
    assert q==p
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    return Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=MAX_SUBDIVISION)
def Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level,MAXLEVEL=2):
    """Optimized subdivision function with iterative approach for leaf level"""
    n_basis = (p+1)*(q+1)
    
    if level == MAXLEVEL:
        # At maximum level, perform Gauss quadrature on 4 subdomains
        # Each GaussQuadrature call now includes the correct Jacobian for its subdomain
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        # Accumulate directly - the Jacobian is already included in each GaussQuadrature call
        K = np.zeros((n_basis, n_basis))
        F = np.zeros(n_basis)
        
        # Process all 4 subdomains
        for sub_x1, sub_x2, sub_y1, sub_y2 in [
            (x1, halfx, y1, halfy),
            (halfx, x2, y1, halfy),
            (x1, halfx, halfy, y2),
            (halfx, x2, halfy, y2)
        ]:
            Ks, Fs = GaussQuadrature(model, sub_x1, sub_x2, sub_y1, sub_y2, 
                                   i, j, p, q, knotvector_x, knotvector_y, Bspxi, Bspeta)
            K += Ks
            F += Fs
        
        # No scaling needed - Jacobian is correctly handled in GaussQuadrature
        return K, F
    else:
        # Recursive subdivision - use in-place accumulation
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        K = np.zeros((n_basis, n_basis))
        F = np.zeros(n_basis)
        
        # Process 4 quadrants recursively
        for sub_x1, sub_x2, sub_y1, sub_y2 in [
            (x1, halfx, y1, halfy),
            (x1, halfx, halfy, y2),
            (halfx, x2, y1, halfy),
            (halfx, x2, halfy, y2)
        ]:
            Kret, Fret = Subdivide(model, sub_x1, sub_x2, sub_y1, sub_y2,
                                 i, j, knotvector_x, knotvector_y, p, q, 
                                 Bspxi, Bspeta, level + 1, MAXLEVEL)
            K += Kret
            F += Fret
        
        # No scaling needed - Jacobian is correctly handled in child calls
        return K, F

def GaussQuadrature(model,x1,x2,y1,y2,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta):
    """Fully vectorized Gauss quadrature - eliminates all Python loops over Gauss points"""
    # Use cached Gauss points
    gaussP_x, gaussP_y, gauss_weights, num_gauss_points = _get_gauss_points(p)
    
    n_basis = (p+1)*(q+1)
    K = np.zeros((n_basis, n_basis))
    F = np.zeros(n_basis)
    
    # Transform Gauss points to physical coordinates
    # Jacobian of the coordinate transformation from [-1,1]x[-1,1] to [x1,x2]x[y1,y2]
    Jxi = (x2 - x1) / 2
    Jeta = (y2 - y1) / 2
    Jacobian = Jxi * Jeta  # Determinant of the Jacobian matrix
    
    xi = Jxi * gaussP_x + (x2+x1)/2
    eta = Jeta * gaussP_y + (y2+y1)/2

    # Get distance function and derivatives (already vectorized)
    d_,dx_,dy_ = mesh.distance_with_derivative_vect_trasformed(xi,eta,model)
    
    # Convert to numpy if needed and flatten
    if hasattr(d_, 'detach'):
        d_np = d_.detach().numpy().flatten()
        dx_np = dx_.detach().numpy().flatten()
        dy_np = dy_.detach().numpy().flatten()
    else:
        d_np = np.asarray(d_).flatten()
        dx_np = np.asarray(dx_).flatten()
        dy_np = np.asarray(dy_).flatten()
    
    # Pre-compute valid gauss points (where d >= 0)
    valid_mask = d_np >= 0
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return K, F
    
    # Extract valid data only once
    d_valid = d_np[valid_mask]
    dx_valid = dx_np[valid_mask]
    dy_valid = dy_np[valid_mask]
    weights_valid = gauss_weights[valid_mask]
    xi_valid = xi[valid_mask]
    eta_valid = eta[valid_mask]
    
    # Compute B-spline values for all valid points at once
    # Shape: (n_valid, n_basis_functions)
    bxi_all = Bspxi.collmat(xi_valid)
    beta_all = Bspeta.collmat(eta_valid) 
    dbdxi_all = Bspxi.collmat(xi_valid, 1)
    dbdeta_all = Bspeta.collmat(eta_valid, 1)
    
    # Ensure 2D shape: (n_valid, n_basis_x) and (n_valid, n_basis_y)
    if bxi_all.ndim == 1:
        bxi_all = bxi_all.reshape(1, -1)
    if beta_all.ndim == 1:
        beta_all = beta_all.reshape(1, -1)
    if dbdxi_all.ndim == 1:
        dbdxi_all = dbdxi_all.reshape(1, -1)
    if dbdeta_all.ndim == 1:
        dbdeta_all = dbdeta_all.reshape(1, -1)
    
    # Get basis function indices (absolute indices into the basis function arrays)
    x_indices = np.arange(i-p, i+1)
    y_indices = np.arange(j-q, j+1)
    
    # Extract relevant basis functions: shape (n_valid, p+1) and (n_valid, q+1)
    bxi_valid = bxi_all[:, x_indices]
    beta_valid = beta_all[:, y_indices]
    dbdxi_valid = dbdxi_all[:, x_indices]
    dbdeta_valid = dbdeta_all[:, y_indices]
    
    # Build full tensor product basis functions: shape (n_valid, p+1, q+1)
    # N[g,i,j] = bxi[g,i] * beta[g,j]
    N_all = bxi_valid[:, :, np.newaxis] * beta_valid[:, np.newaxis, :]  # (n_valid, p+1, q+1)
    dN_xi_all = dbdxi_valid[:, :, np.newaxis] * beta_valid[:, np.newaxis, :]  # (n_valid, p+1, q+1)
    dN_eta_all = bxi_valid[:, :, np.newaxis] * dbdeta_valid[:, np.newaxis, :]  # (n_valid, p+1, q+1)
    
    # Reshape to (n_valid, n_basis) for easier manipulation
    N_flat = N_all.reshape(n_valid, n_basis)
    dN_xi_flat = dN_xi_all.reshape(n_valid, n_basis)
    dN_eta_flat = dN_eta_all.reshape(n_valid, n_basis)
    
    # Compute corrected gradients: (n_valid, n_basis)
    # corr_xi = dN_xi * d + dx * N
    # corr_eta = dN_eta * d + dy * N
    corr_xi = dN_xi_flat * d_valid[:, np.newaxis] + dx_valid[:, np.newaxis] * N_flat
    corr_eta = dN_eta_flat * d_valid[:, np.newaxis] + dy_valid[:, np.newaxis] * N_flat
    
    # Pre-compute function values (already vectorized)
    f_values = load_function_vectorized(xi_valid, eta_valid)
    dirichlet_values = dirichletBoundary_vectorized(xi_valid, eta_valid)
    dirichlet_dx = dirichletBoundaryDerivativeX_vectorized(xi_valid, eta_valid)
    dirichlet_dy = dirichletBoundaryDerivativeY_vectorized(xi_valid, eta_valid)
    
    # Ensure proper shapes
    if hasattr(f_values, 'flatten'):
        f_values = np.asarray(f_values).flatten()
        dirichlet_values = np.asarray(dirichlet_values).flatten()
        dirichlet_dx = np.asarray(dirichlet_dx).flatten()
        dirichlet_dy = np.asarray(dirichlet_dy).flatten()
    
    # Vectorized K matrix assembly using einsum
    # K[m,n] = sum_g weight[g] * Jacobian * (corr_xi[g,m]*corr_xi[g,n] + corr_eta[g,m]*corr_eta[g,n])
    weighted_corr_xi = corr_xi * weights_valid[:, np.newaxis]  # (n_valid, n_basis)
    weighted_corr_eta = corr_eta * weights_valid[:, np.newaxis]  # (n_valid, n_basis)
    
    # Use einsum for outer product summation - this is the key optimization
    # Include Jacobian from coordinate transformation
    K = (np.einsum('gi,gj->ij', weighted_corr_xi, corr_xi) + np.einsum('gi,gj->ij', weighted_corr_eta, corr_eta)) * Jacobian
    
    # Vectorized F vector assembly
    # F[m] = sum_g weight[g] * Jacobian * (f * d * N[m] + corr_xi[m]*(dx*dirichlet + d*ddir_x) 
    #                          + corr_eta[m]*(dy*dirichlet + d*ddir_y) - ddir_x*corr_xi[m] - ddir_y*corr_eta[m])
    Ni_corr = d_valid[:, np.newaxis] * N_flat  # (n_valid, n_basis)
    
    term1 = f_values[:, np.newaxis] * Ni_corr  # (n_valid, n_basis)
    term2_xi = corr_xi * (dx_valid * dirichlet_values + d_valid * dirichlet_dx)[:, np.newaxis]
    term2_eta = corr_eta * (dy_valid * dirichlet_values + d_valid * dirichlet_dy)[:, np.newaxis]
    term3 = dirichlet_dx[:, np.newaxis] * corr_xi + dirichlet_dy[:, np.newaxis] * corr_eta
    
    F_contrib = (term1 + term2_xi + term2_eta - term3) * weights_valid[:, np.newaxis]
    F = np.sum(F_contrib, axis=0) * Jacobian
    
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

# Pre-allocated corner points tensor for element classification
_CORNER_POINTS_BUFFER = torch.zeros((4, 2), dtype=torch.float64)

# Cache for zero matrices/vectors by size
_ZERO_CACHE = {}

def _get_zero_arrays(n_basis):
    """Get cached zero arrays for outer elements"""
    if n_basis not in _ZERO_CACHE:
        _ZERO_CACHE[n_basis] = (np.zeros((n_basis, n_basis), dtype=np.float64), np.zeros(n_basis, dtype=np.float64))
    # Return copies to avoid mutation issues
    K_zero, F_zero = _ZERO_CACHE[n_basis]
    return K_zero.copy(), F_zero.copy()

def elementChoose(model,p,q,knotvector_x, knotvector_y,i,j,etype=None):
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    
    # Use pre-allocated buffer to avoid allocations
    _CORNER_POINTS_BUFFER[0, 0] = x1
    _CORNER_POINTS_BUFFER[0, 1] = y1
    _CORNER_POINTS_BUFFER[1, 0] = x2
    _CORNER_POINTS_BUFFER[1, 1] = y1
    _CORNER_POINTS_BUFFER[2, 0] = x1
    _CORNER_POINTS_BUFFER[2, 1] = y2
    _CORNER_POINTS_BUFFER[3, 0] = x2
    _CORNER_POINTS_BUFFER[3, 1] = y2
    
    with torch.no_grad():  # Disable gradient tracking for classification
        distances = model(_CORNER_POINTS_BUFFER)
    
    # Fast element classification
    d_flat = distances.view(-1)
    min_d = d_flat.min().item()
    max_d = d_flat.max().item()
    
    n_basis = (p+1)*(q+1)
    
    if min_d >= 0:  # All inside - inner element
        if etype is not None: 
            etype["inner"] += 1
        Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
        Ke, Fe = element(model, p, q, knotvector_x, knotvector_y, i, j, Bspxi, Bspeta)
    elif max_d < 0:  # All outside - outer element
        if etype is not None: 
            etype["outer"] += 1
        Ke, Fe = _get_zero_arrays(n_basis)
    else:  # Mixed - boundary element
        if etype is not None: 
            etype["boundary"] += 1
        Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
        Ke, Fe = boundaryElement(model, p, q, knotvector_x, knotvector_y, i, j, Bspxi, Bspeta)
    
    if etype is not None: 
        return Ke, Fe, etype
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
    points = torch.tensor(np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]]),dtype=TORCH_DTYPE)
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
# Pre-computed assembly index cache
_ASSEMBLY_INDEX_CACHE = {}

def _get_assembly_indices(elemx, elemy, p, q, xDivision, yDivision):
    """Cache assembly indices to avoid recomputation"""
    key = (elemx, elemy, p, q, xDivision, yDivision)
    if key in _ASSEMBLY_INDEX_CACHE:
        return _ASSEMBLY_INDEX_CACHE[key]
    
    stride = xDivision + p + 1
    base = (elemx - p) * stride + (elemy - q)
    
    # Vectorized index computation
    ii = np.arange(p + 1)
    jj = np.arange(q + 1)
    idxs = (base + ii[:, np.newaxis] * stride + jj).flatten()
    
    _ASSEMBLY_INDEX_CACHE[key] = idxs
    return idxs

def assembly(K, F, Ke, Fe, elemx, elemy, p, q, xDivision, yDivision):
    """Optimized assembly using NumPy advanced indexing"""
    # Get cached indices
    idxs = _get_assembly_indices(elemx, elemy, p, q, xDivision, yDivision)
    
    # Vectorized matrix assembly using outer indexing
    idx_mesh = np.ix_(idxs, idxs)
    K[idx_mesh] += Ke
    
    # Vectorized vector assembly
    F[idxs] += Fe
    
    return K, F

def solveWeak(K, F):
    """Optimized linear solver with efficient zero-row detection"""
    # Use np.any for faster zero-row detection (checks for non-zero)
    non_zero_rows = np.any(K != 0, axis=1)
    
    if not np.any(non_zero_rows):
        return np.zeros(len(F))
    
    # For symmetric case, zero rows == zero cols
    K_reduced = K[non_zero_rows][:, non_zero_rows]
    F_reduced = F[non_zero_rows]
    
    u = np.zeros(len(F))
    #print("condition number of reduced K:", np.linalg.cond(K_reduced))
    # Try scipy.linalg.solve for potentially better performance (uses LAPACK)
    try:
        from scipy import linalg
        u_reduced = linalg.solve(K_reduced, F_reduced, assume_a='sym')
    except ImportError:
        # Fall back to numpy if scipy not available
        u_reduced = np.linalg.solve(K_reduced, F_reduced)
    print("Max error:", np.max(np.abs(K_reduced @ u_reduced - F_reduced)))
    u[non_zero_rows] = u_reduced
    return u


def classifyAllElements(model, p, q, knotvector_x, knotvector_y, xDivision):
    """Batch classify all elements with a single neural network call.
    
    Returns:
        element_types: dict with lists of (elemx, elemy) tuples for each type
    """
    # Build all corner points for all elements
    n_elem = (xDivision + 1) * (xDivision + 1)
    all_corners = []
    elem_coords = []
    
    for elemx in range(p, p + xDivision + 1):
        for elemy in range(q, q + xDivision + 1):
            x1, x2 = knotvector_x[elemx], knotvector_x[elemx + 1]
            y1, y2 = knotvector_y[elemy], knotvector_y[elemy + 1]
            all_corners.extend([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            elem_coords.append((elemx, elemy))
    
    # Single batched neural network call
    corners_tensor = torch.tensor(all_corners, dtype=TORCH_DTYPE)
    with torch.no_grad():
        all_distances = model(corners_tensor).view(-1, 4)  # (n_elem, 4)
    
    # Classify elements
    min_d = all_distances.min(dim=1).values.numpy()
    max_d = all_distances.max(dim=1).values.numpy()
    
    inner_mask = min_d >= 0
    outer_mask = max_d < 0
    
    element_types = {"inner": [], "outer": [], "boundary": []}
    
    for idx, (elemx, elemy) in enumerate(elem_coords):
        if inner_mask[idx]:
            element_types["inner"].append((elemx, elemy))
        elif outer_mask[idx]:
            element_types["outer"].append((elemx, elemy))
        else:
            element_types["boundary"].append((elemx, elemy))
    
    return element_types


def processAllElements(model, p, q, knotvector_x, knotvector_y, xDivision, yDivision, K, F):
    """Process all elements with optimized batch classification.
    
    Returns:
        K, F: Updated stiffness matrix and load vector
        etype: Element type counts
    """
    # Batch classify all elements first
    element_types = classifyAllElements(model, p, q, knotvector_x, knotvector_y, xDivision)
    
    # Get cached Bspline objects once
    Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
    n_basis = (p + 1) * (q + 1)
    
    etype = {
        "inner": len(element_types["inner"]),
        "outer": len(element_types["outer"]),
        "boundary": len(element_types["boundary"])
    }
    
    total_elements = etype["inner"] + etype["boundary"]
    
    # Create progress bar for all non-outer elements
    with tqdm(total=total_elements, desc="Processing elements", unit="elem") as pbar:
        # Process inner elements (no subdivision needed)
        for elemx, elemy in element_types["inner"]:
            Ke, Fe = element(model, p, q, knotvector_x, knotvector_y, elemx, elemy, Bspxi, Bspeta)
            K, F = assembly(K, F, Ke, Fe, elemx, elemy, p, q, xDivision, yDivision)
            pbar.update(1)
        
        # Update description for boundary elements (more expensive)
        pbar.set_description("Processing boundary elements")
        
        # Process boundary elements (need full subdivision)
        for elemx, elemy in element_types["boundary"]:
            Ke, Fe = boundaryElement(model, p, q, knotvector_x, knotvector_y, elemx, elemy, Bspxi, Bspeta)
            K, F = assembly(K, F, Ke, Fe, elemx, elemy, p, q, xDivision, yDivision)
            pbar.update(1)
    
    return K, F, etype

