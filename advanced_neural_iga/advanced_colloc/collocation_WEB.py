"""
Collocation with WEB-splines (Weighted Extended B-splines) for 2D Poisson equation.

This is a Python port of the MATLAB CwBS-Programs:
    Apprich, C., Höllig, K., Hörner, J., Reif, U.
    Original reference: "Collocation with WEB-Splines" (CwBS-Programs Version 1.0)

The algorithm solves:
    -Δu = f  on D: w(x,y) > 0
    u = 0    on ∂D (Dirichlet boundary, implicitly enforced by weight function)

Domain D is defined implicitly via a weight function w(x,y) > 0.

Based on:
    Höllig, K., Reif, U., Wipper, J. (2001). "Weighted Extended B-Spline 
    Approximation of Dirichlet Problems." Mathematics of Computation, 70(235), 51-63.

Author: Neural IGA Research
Date: 2026-01-26
"""

import importlib

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as dense_solve
import time
from typing import Callable, Tuple, Dict, Optional, Union
import torch
import PDE_testcases
import inhomogenous_boundary as ihbnd
#from PDE_testcases import *
#import FEM



# Use float64 for better numerical accuracy
NP_DTYPE = np.float64
TORCH_DTYPE = torch.float64

# =============================================================================
# FUNCTION CASE CONFIGURATION (compatible with FEM.py)
# =============================================================================
# FUNCTION_CASE defines the test problem:
#   1: u = x*(x^2 + y^2 - 1), f = -8x (circle domain)
#   2: u = cos(pi/2 * (x^2+y^2)), f = complex (circle domain)
#   3: u = x*(x^2 + y^2 - 1) + 2, f = -8x (circle domain with offset)
#   4: u = x*(x^2 + y^2 - 1) + x + 2y, f = -8x (circle domain with linear)
#   5: u = sin(2*pi*x)*sin(2*pi*y), f = 8*pi^2*sin(...) (L-shape domain)
#   7: u = sin(pi*(x^2+y^2)), f = complex (double circle domain)
#   0: u = exp(w) - 1 (default WEB-spline example, works for any domain)

#FUNCTION_CASE = 3  # Default: use exp(w)-1 exact solution



# =============================================================================
# B-SPLINE EVALUATION FUNCTIONS
# =============================================================================

def bspline_evaluate(t: np.ndarray, n: int) -> np.ndarray:
    """
    Evaluate the standard B-spline of degree n at points t.
    
    The B-spline is the cardinal B-spline with knots at integers 0, 1, ..., n+1.
    Support: [0, n+1]
    
    Parameters:
    -----------
    t : ndarray
        Evaluation points
    n : int
        B-spline degree (order n+1)
    
    Returns:
    --------
    b : ndarray
        B-spline values at t
    """
    t = np.asarray(t, dtype=NP_DTYPE)
    
    if n == 0:
        b = np.zeros_like(t)
        mask = (t >= 0) & (t < 1)
        b[mask] = 1.0
    elif n == 1:
        b = np.zeros_like(t)
        mask1 = (t >= 0) & (t < 1)
        b[mask1] = t[mask1]
        mask2 = (t >= 1) & (t < 2)
        b[mask2] = 2 - t[mask2]
    else:
        # Recursion: B_n(t) = (t/n)*B_{n-1}(t) + ((n+1-t)/n)*B_{n-1}(t-1)
        b = (t / n) * bspline_evaluate(t, n - 1) + ((n + 1 - t) / n) * bspline_evaluate(t - 1, n - 1)
    
    return b


def bspline_derivative(t: np.ndarray, n: int, order: int = 1) -> np.ndarray:
    """
    Evaluate the derivative of a B-spline of degree n at points t.
    
    Uses the B-spline derivative formula:
        d/dt B_n(t) = B_{n-1}(t) - B_{n-1}(t-1)
        d^2/dt^2 B_n(t) = B_{n-2}(t) - 2*B_{n-2}(t-1) + B_{n-2}(t-2)
    
    Parameters:
    -----------
    t : ndarray
        Evaluation points
    n : int
        B-spline degree
    order : int
        Derivative order (1 or 2)
    
    Returns:
    --------
    db : ndarray
        Derivative values at t
    """
    t = np.asarray(t, dtype=NP_DTYPE)
    
    if order == 1:
        if n < 1:
            return np.zeros_like(t)
        return bspline_evaluate(t, n - 1) - bspline_evaluate(t - 1, n - 1)
    elif order == 2:
        if n < 2:
            return np.zeros_like(t)
        return bspline_evaluate(t, n - 2) - 2 * bspline_evaluate(t - 1, n - 2) + bspline_evaluate(t - 2, n - 2)
    else:
        raise ValueError(f"Derivative order {order} not supported (use 1 or 2)")


# =============================================================================
# EXTENSION COEFFICIENT COMPUTATION
# =============================================================================

def extension_coefficient_1d(I_indices: np.ndarray, j: int) -> np.ndarray:
    """
    Compute 1D Lagrange extension coefficients e_{i,j} for i in I.
    
    For the Lagrange polynomial that interpolates f at points I and evaluates at j:
        e_{i,j} = ∏_{k in I, k≠i} (j - k) / (i - k)
    
    This is equivalent to the Lagrange basis polynomial L_i(j).
    
    Parameters:
    -----------
    I_indices : ndarray
        Array of inner indices (typically [α, α+1, ..., α+n] for degree n)
    j : int
        Outer index to extend to
    
    Returns:
    --------
    E : ndarray
        Extension coefficients for each i in I
    """
    I = np.asarray(I_indices, dtype=NP_DTYPE)
    n = len(I) - 1
    E = np.ones(n + 1, dtype=NP_DTYPE)
    
    for i_idx in range(n + 1):
        i_val = I[i_idx]
        for k_idx in range(n + 1):
            if k_idx != i_idx:
                k_val = I[k_idx]
                E[i_idx] *= (j - k_val) / (i_val - k_val)
    
    return E


# =============================================================================
# PRECOMPUTE EXTENSION AND B-SPLINE DATA
# =============================================================================

def compute_collocation_data(n: int, J_MAX: int = 16) -> Dict:
    """
    Precompute extension coefficients and B-spline values for degree n.
    
    This is the Python equivalent of collocation_data_2d.m.
    
    Parameters:
    -----------
    n : int
        B-spline degree
    J_MAX : int
        Maximal distance of arrays of inner B-spline centers
    
    Returns:
    --------
    CD : dict
        Dictionary containing:
        - 'E': Extension coefficients array shape ((n+1)^2, dim, dim, 2, 2)
        - 'b00', 'b10', 'b01', 'b20', 'b02': B-spline product values
    """
    # Bandwidth: 1+2*bw points within a B-spline support
    bw = n // 2  # floor(n/2) - even n+1, odd n
    
    # Extension coefficients
    dim = int(J_MAX + n / 2) + 1
    
    # 1D extension coefficients first
    t = np.arange(n + 1, dtype=NP_DTYPE)
    EU = np.zeros((n + 1, dim, 2), dtype=NP_DTYPE)
    
    for i in range(n + 1):
        for j in range(-J_MAX, n + J_MAX + 1):
            # Sign indicator D0 and distance D1
            if (j - n / 2) >= 0:
                D0 = 1  # Index 1 in Python (0-based would be 1)
            else:
                D0 = 0  # Index 0
            D1 = int(abs(j - n / 2))
            
            if D1 < dim:
                # Lagrange coefficient: evaluate at j the polynomial that is 1 at t[i] and 0 elsewhere
                coeff = extension_coefficient_1d(t, j)
                EU[i, D1, D0] = round(coeff[i])  # Round to get exact integer coefficients
    
    # Build 2D tensor product of extension coefficients
    E = np.zeros(((n + 1) ** 2, dim, dim, 2, 2), dtype=NP_DTYPE)
    for j1 in range(dim):
        for j2 in range(dim):
            for s1 in range(2):
                for s2 in range(2):
                    E[:, j1, j2, s1, s2] = np.outer(EU[:, j1, s1], EU[:, j2, s2]).flatten()
    
    # B-spline values at evaluation points
    t_eval = np.arange(-bw, bw + 1) + (n + 1) / 2  # Points within support
    
    b = bspline_evaluate(t_eval, n)
    db = bspline_derivative(t_eval, n, order=1)
    ddb = bspline_derivative(t_eval, n, order=2)
    
    # Tensor products of B-splines and their derivatives
    b00 = np.outer(b, b)   # B_i(x) * B_j(y)
    b10 = np.outer(db, b)  # B'_i(x) * B_j(y)
    b01 = np.outer(b, db)  # B_i(x) * B'_j(y)
    b20 = np.outer(ddb, b) # B''_i(x) * B_j(y)
    b02 = np.outer(b, ddb) # B_i(x) * B''_j(y)
    
    return {
        'E': E,
        'b00': b00,
        'b10': b10,
        'b01': b01,
        'b20': b20,
        'b02': b02,
        'n': n,
        'bw': bw,
        'dim': dim,
    }


# =============================================================================
# WEIGHT FUNCTION INTERFACE
# =============================================================================

class WeightFunction:
    """
    Base class for weight functions defining the domain D: w(x,y) > 0.
    
    Subclasses must implement __call__ to return (w, wx, wy, wxx, wyy).
    """
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Evaluate weight function and its derivatives.
        
        Parameters:
        -----------
        x, y : ndarray
            Evaluation points (same shape)
        
        Returns:
        --------
        w : ndarray
            Weight function value
        wx, wy : ndarray
            First derivatives
        wxx, wyy : ndarray
            Second derivatives
        """
        raise NotImplementedError("Subclasses must implement __call__")

class ShovelWeightFunction(WeightFunction):
    """
    Weight function for a 'shovel' shaped domain.
    """
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        x = np.asarray(x, dtype=NP_DTYPE)
        y = np.asarray(y, dtype=NP_DTYPE)
        
        p = 2 * x - 1
        q = 2.25 * y - 0.25
        
        w = 1 - p ** 2 - (q + p ** 2 - 1) ** 2
        wx = -8 * p * q - 8 * p ** 3 + 4 * p
        wy = -4.5 * (q + p ** 2 - 1)
        wxx = -16 * q - 48 * p ** 2 + 8
        wyy = np.full_like(y, -10.125)
        
        return w, wx, wy, wxx, wyy


class NeuralWeightFunction(WeightFunction):
    """
    Weight function using a neural network SDF model.
    
    The SDF is scaled/shifted to define w(x,y) > 0 inside the domain.
    """
    
    def __init__(self, model: torch.nn.Module, domain: Dict[str, float] = None, 
                 transform: Optional[str] = None, tang: float = 1.0):
        """
        Parameters:
        -----------
        model : torch.nn.Module
            Neural network that outputs SDF values
        domain : dict
            Physical domain bounds {'x1': ..., 'x2': ..., 'y1': ..., 'y2': ...}
            Points in [0,1]^2 are mapped to this domain for model evaluation.
            If None, assumes model works on [0,1]^2 directly.
        transform : str, optional
            Transform to apply to SDF output: 'sigmoid', 'tanh', 'logarithmic', 
            'exponential', 'trapezoid', or None (no transform)
        tang : float
            Tangent parameter for trapezoid transform (default: 1.0)
        """
        self.model = model
        self.domain = domain
        self.transform = transform
        self.tang = tang
        self.model.eval()
        
        # Try to get device from model
        self.device = torch.device('cpu')
        try:
            params = list(model.parameters())
            if len(params) > 0:
                self.device = params[0].device
        except:
            pass
    
    def _transform_coords(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform from [0,1]^2 to physical domain."""
        if self.domain is None:
            return x, y
        
        # Map [0,1] -> [domain_min, domain_max]
        x_phys = self.domain['x1'] + x * (self.domain['x2'] - self.domain['x1'])
        y_phys = self.domain['y1'] + y * (self.domain['y2'] - self.domain['y1'])
        return x_phys, y_phys
    
    def transform_to_physical(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Public method to transform grid coordinates [0,1]^2 to physical domain.
        
        Use this to transform xB, yB before passing to load_function/solution_function
        when those functions are defined in the physical domain.
        
        Parameters:
        -----------
        x, y : ndarray
            Grid coordinates in [0,1]^2
        
        Returns:
        --------
        x_phys, y_phys : ndarray
            Physical coordinates in [domain['x1'], domain['x2']] x [domain['y1'], domain['y2']]
        """
        return self._transform_coords(np.asarray(x, dtype=NP_DTYPE), 
                                       np.asarray(y, dtype=NP_DTYPE))
    
    def _apply_transform(self, w_tensor: torch.Tensor, wx_tensor: torch.Tensor, 
                        wy_tensor: torch.Tensor, wxx_tensor: torch.Tensor, 
                        wyy_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply transformation to SDF and its derivatives using chain rule.
        
        For a transform g(w), we have:
        - w_new = g(w)
        - wx_new = g'(w) * wx
        - wy_new = g'(w) * wy
        - wxx_new = g''(w) * wx^2 + g'(w) * wxx
        - wyy_new = g''(w) * wy^2 + g'(w) * wyy
        """
        if self.transform is None:
            return w_tensor, wx_tensor, wy_tensor, wxx_tensor, wyy_tensor
        
        if self.transform == 'sigmoid':
            # g(w) = 1 / (1 + exp(-w))
            # g'(w) = g(w) * (1 - g(w))
            # g''(w) = g'(w) * (1 - 2*g(w))
            exp_neg_w = torch.exp(-w_tensor)
            g_w = 1.0 / (1.0 + exp_neg_w)
            g_prime = g_w * (1.0 - g_w)
            g_double_prime = g_prime * (1.0 - 2.0 * g_w)
            
            w_new = g_w
            wx_new = g_prime * wx_tensor
            wy_new = g_prime * wy_tensor
            wxx_new = g_double_prime * wx_tensor ** 2 + g_prime * wxx_tensor
            wyy_new = g_double_prime * wy_tensor ** 2 + g_prime * wyy_tensor
            
        else:
            raise ValueError(f"Unknown transform: {self.transform}. "
                           f"Use 'sigmoid', 'tanh', 'logarithmic', 'exponential', 'trapezoid', or None")
        
        return w_new, wx_new, wy_new, wxx_new, wyy_new
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        x = np.asarray(x, dtype=NP_DTYPE)
        y = np.asarray(y, dtype=NP_DTYPE)
        original_shape = x.shape
        
        # Transform to physical coordinates
        x_phys, y_phys = self._transform_coords(x.flatten(), y.flatten())
        
        # Prepare tensor input with gradient tracking
        xy = np.stack([x_phys, y_phys], axis=-1)
        xy_tensor = torch.tensor(xy, dtype=TORCH_DTYPE, device=self.device, requires_grad=True)
        
        # Forward pass
        w_tensor = self.model(xy_tensor)
        if w_tensor.dim() > 1:
            w_tensor = w_tensor.squeeze(-1)
        
        # First derivatives via autograd
        grad_w = torch.autograd.grad(
            outputs=w_tensor.sum(),
            inputs=xy_tensor,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        wx_tensor = grad_w[:, 0]
        wy_tensor = grad_w[:, 1]
        
        # Second derivatives
        zero_template = torch.zeros_like(xy_tensor)

        if wx_tensor.requires_grad:
            try:
                grad_wx = torch.autograd.grad(
                    outputs=wx_tensor.sum(),
                    inputs=xy_tensor,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                grad_wx = None
        else:
            grad_wx = None

        if wy_tensor.requires_grad:
            try:
                grad_wy = torch.autograd.grad(
                    outputs=wy_tensor.sum(),
                    inputs=xy_tensor,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                grad_wy = None
        else:
            grad_wy = None

        if grad_wx is None:
            grad_wx = zero_template

        if grad_wy is None:
            grad_wy = zero_template
        
        wxx_tensor = grad_wx[:, 0]
        wyy_tensor = grad_wy[:, 1]
        
        # Apply transformation if specified (before domain scaling)
        w_tensor, wx_tensor, wy_tensor, wxx_tensor, wyy_tensor = self._apply_transform(
            w_tensor, wx_tensor, wy_tensor, wxx_tensor, wyy_tensor
        )
        
        # Convert to numpy
        w = w_tensor.detach().cpu().numpy().reshape(original_shape)
        wx = wx_tensor.detach().cpu().numpy().reshape(original_shape)
        wy = wy_tensor.detach().cpu().numpy().reshape(original_shape)
        wxx = wxx_tensor.detach().cpu().numpy().reshape(original_shape)
        wyy = wyy_tensor.detach().cpu().numpy().reshape(original_shape)
        
        # Scale derivatives if domain transform was applied
        if self.domain is not None:
            scale_x = self.domain['x2'] - self.domain['x1']
            scale_y = self.domain['y2'] - self.domain['y1']
            wx *= scale_x
            wy *= scale_y
            wxx *= scale_x ** 2
            wyy *= scale_y ** 2
        
        return w, wx, wy, wxx, wyy


# =============================================================================
# COORDINATE TRANSFORMATION UTILITIES
# =============================================================================

def create_domain_transformer(domain: Dict[str, float]):
    """
    Create coordinate transformation functions for a given physical domain.
    
    Use this to wrap exact solution and derivative functions when the domain
    is not [0,1]^2.
    
    Parameters:
    -----------
    domain : dict
        Physical domain bounds {'x1': ..., 'x2': ..., 'y1': ..., 'y2': ...}
    
    Returns:
    --------
    transformer : object with methods:
        - to_physical(x, y): transform [0,1]^2 -> physical domain
        - wrap_function(f): wrap a function f(x,y) to accept grid coords
        - wrap_derivative_x(df_dx): wrap du/dx with chain rule scaling
        - wrap_derivative_y(df_dy): wrap du/dy with chain rule scaling
        - scale_x, scale_y: domain scale factors
    """
    scale_x = domain['x2'] - domain['x1']
    scale_y = domain['y2'] - domain['y1']
    
    class DomainTransformer:
        def __init__(self):
            self.scale_x = scale_x
            self.scale_y = scale_y
            self.domain = domain
        
        def to_physical(self, x, y):
            """Transform grid [0,1]^2 to physical domain."""
            x_phys = domain['x1'] + np.asarray(x) * scale_x
            y_phys = domain['y1'] + np.asarray(y) * scale_y
            return x_phys, y_phys
        
        def wrap_function(self, f):
            """Wrap function f(x,y) to accept grid coordinates."""
            def wrapped(x, y):
                x_phys, y_phys = self.to_physical(x, y)
                return f(x_phys, y_phys)
            return wrapped
        
        def wrap_derivative_x(self, df_dx):
            """Wrap du/dx with chain rule: du/dx_grid = du/dx_phys * scale_x."""
            def wrapped(x, y):
                x_phys, y_phys = self.to_physical(x, y)
                return df_dx(x_phys, y_phys) * scale_x
            return wrapped
        
        def wrap_derivative_y(self, df_dy):
            """Wrap du/dy with chain rule: du/dy_grid = du/dy_phys * scale_y."""
            def wrapped(x, y):
                x_phys, y_phys = self.to_physical(x, y)
                return df_dy(x_phys, y_phys) * scale_y
            return wrapped
    
    return DomainTransformer()


# =============================================================================
# ARRAY TO SPARSE MATRIX CONVERSION
# =============================================================================

def array_to_matrix(G: np.ndarray) -> sparse.csr_matrix:
    """
    Convert a banded array G to a sparse matrix.
    
    This is the Python equivalent of array2matrix in collocation_2d.m.
    
    Parameters:
    -----------
    G : ndarray
        4D array of shape (dim, dim, 2*bw+1, 2*bw+1)
        G[k1, k2, s1, s2] contains the matrix entry at row (k1, k2)
        and column (k1 + (s1-bw), k2 + (s2-bw))
    
    Returns:
    --------
    GM : sparse.csr_matrix
        Sparse collocation matrix of shape (dim^2, dim^2)
    """
    dim = G.shape[0]
    bw = (G.shape[2] - 1) // 2
    c = dim ** 2
    
    # Build index arrays following MATLAB logic but adapted for Python row-major ordering
    # MATLAB uses column-major (Fortran) ordering: linear_index = k1 + k2*dim
    # Python uses row-major (C) ordering: linear_index = k1*dim + k2
    # MATLAB: [K1,K2]=ndgrid(-bw:bw)
    k1_range = np.arange(-bw, bw + 1)
    k2_range = np.arange(-bw, bw + 1)
    K1, K2 = np.meshgrid(k1_range, k2_range, indexing='ij')
    
    # MATLAB: I = K1 + dim*K2 (column offset from diagonal in column-major)
    # Python: I = K1*dim + K2 (offset from diagonal in row-major)
    I_offset = K1 * dim + K2
    
    # Create full index arrays
    # I1: row indices (all c rows for each (s1,s2) combination)
    # I2: column indices (row index + offset)
    n_offsets = (2 * bw + 1) ** 2
    
    # I1 has shape (c, n_offsets): each row i has the same row index for all offsets
    I1 = np.tile(np.arange(c), (n_offsets, 1)).T
    
    # I2 = I1 + offset for each (s1, s2)
    I2 = I1 + I_offset.flatten()[np.newaxis, :]
    
    # Reshape I2 to match G dimensions for boundary handling
    I2_shaped = I2.reshape(dim, dim, 2 * bw + 1, 2 * bw + 1)
    
    # MATLAB boundary handling:
    # for k=1:bw: I2(k,:,bw+1-k,:)=-1; I2(dim+1-k,:,bw+1+k,:)=-1
    for k in range(1, bw + 1):
        # I2(k,:,bw+1-k,:)=-1 in MATLAB (1-based)
        # In Python (0-based): I2[k-1, :, bw-k, :] = -1
        I2_shaped[k - 1, :, bw + 1 - k - 1, :] = -1  # bw+1-k in MATLAB -> bw-k in Python
        # I2(dim+1-k,:,bw+1+k,:)=-1 in MATLAB
        # In Python: I2[dim-k, :, bw+k, :] = -1
        I2_shaped[dim - k, :, bw + k, :] = -1
    
    # Flatten back
    I2 = I2_shaped.reshape(c, n_offsets)
    
    # Find valid indices: I2 > 0 and I2 <= c (MATLAB uses 1-based)
    # In Python: I2 >= 0 and I2 < c
    valid = (I2 >= 0) & (I2 < c)
    
    # Get row, col, data for valid entries
    rows = []
    cols = []
    data = []
    
    for offset_idx in range(n_offsets):
        s1 = offset_idx // (2 * bw + 1)
        s2 = offset_idx % (2 * bw + 1)
        
        G_flat = G[:, :, s1, s2].flatten()
        valid_mask = valid[:, offset_idx]
        
        row_indices = I1[valid_mask, offset_idx]
        col_indices = I2[valid_mask, offset_idx]
        values = G_flat[valid_mask]
        
        # Only add non-zero values
        nonzero_mask = values != 0
        rows.extend(row_indices[nonzero_mask].tolist())
        cols.extend(col_indices[nonzero_mask].tolist())
        data.extend(values[nonzero_mask].tolist())
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(c, c))


# =============================================================================
# MAIN COLLOCATION SOLVER
# =============================================================================

def collocation_2d(
    n: int,
    H: int,
    wfct: WeightFunction,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    CD: Optional[Dict] = None,
    verbose: bool = True,
    domain: Optional[Dict[str, float]] = None,
    solve_method: str = "auto",
    dense_threshold: int = 2000,
    g: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gx: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gy: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gxx: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gyy: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    return_coefficients: bool = False,
    model: Optional[torch.nn.Module] = None,
    function_case: Optional[int] = 11, #!TODO
    bundary_enforcement: Optional[str] = "lifting"
):
    """
    Solve -Δu = f on D: w(x,y) > 0, with u = 0 on ∂D.
    
    Uses WEB-spline collocation method.
    
    Parameters:
    -----------
    n : int
        B-spline degree (2 to 5)
    H : int
        Number of grid intervals (grid width = 1/H)
    wfct : WeightFunction
        Weight function defining domain (w > 0 inside)
    f : callable
        Right-hand side function f(x, y) defined in physical coordinates.
        If domain is specified, coordinates are transformed and Laplacian is scaled.
    CD : dict, optional
        Precomputed collocation data from compute_collocation_data()
    verbose : bool
        Print progress messages
    domain : dict, optional
        Physical domain bounds {'x1': ..., 'x2': ..., 'y1': ..., 'y2': ...}.
        If provided, the grid [0,1]^2 is mapped to this domain for f evaluation,
        and the Laplacian scaling is applied automatically.
        If None (default), f is evaluated directly on grid coordinates [0,1]^2.

    solve_method : str
        Linear solve strategy: 'auto', 'sparse', or 'dense'.
        'auto' uses dense for small systems and sparse direct solve otherwise.
    dense_threshold : int
        If solve_method='auto', use dense solve when mi <= dense_threshold.
    
    Returns:
    --------
    Uxy : ndarray
        Solution values on grid points
    xB, yB : ndarray
        Grid coordinates (always in [0,1]^2)
    con : float
        Condition number estimate of linear system (NaN if dim > 1000)
    dim_sys : int
        Dimension of the linear system
    rtimes : dict
        Computing times for different stages
    """
    rtimes = {'sys': np.nan, 'ext': np.nan, 'sol': np.nan, 'total': np.nan}

    def _early_return(Uxy, xB, yB, con, dim_sys):
        if return_coefficients:
            return Uxy, xB, yB, con, dim_sys, rtimes, None
        return Uxy, xB, yB, con, dim_sys, rtimes
    
    # ---------------------------------------------------------------------
    # Optional non-homogeneous Dirichlet handling via blended ansatz:
    #   u(x,y) = w(x,y) * v_h(x,y) + (1 - w(x,y)) * g(x,y)
    # where v_h is the WEB-spline approximation (coefficients solved by collocation).
    # This reduces to the original homogeneous case when g ≡ 0.
    #
    # Collocation system remains on the unknown part -Δ(w * v_h), with modified RHS:
    #   -Δ(w v_h) = f + Δ((1-w) g)
    #
    # Δ((1-w) g) = -(wxx+wyy) g - 2(wx gx + wy gy) + (1-w)(gxx+gyy)
    # ---------------------------------------------------------------------

    # Set up coordinate transformation and Laplacian scaling.
    #
    # PDE is posed in physical coordinates (x_phys, y_phys):
    #   - (∂²/∂x_phys² + ∂²/∂y_phys²) u = f_phys(x_phys, y_phys).
    # With the affine mapping x_phys = x1 + scale_x * x_grid, the Laplacian becomes:
    #   - (1/scale_x^2) ∂²/∂x_grid² u - (1/scale_y^2) ∂²/∂y_grid² u = f_phys(T(x_grid)).
    #
    # We implement this by:
    #   - mapping all user functions (f, g, ...) to physical coords (no extra scaling),
    #   - assembling the collocation operator with anisotropic factors ax=1/scale_x^2, ay=1/scale_y^2.
    ax = 1.0
    ay = 1.0
    if domain is not None:
        scale_x = domain['x2'] - domain['x1']
        scale_y = domain['y2'] - domain['y1']
        ax = 1.0 / (scale_x ** 2)
        ay = 1.0 / (scale_y ** 2)

        def transform_to_physical(x, y):
            x_phys = domain['x1'] + np.asarray(x, dtype=NP_DTYPE) * scale_x
            y_phys = domain['y1'] + np.asarray(y, dtype=NP_DTYPE) * scale_y
            return x_phys, y_phys

        # Wrap f to transform coordinates (PDE RHS is defined in physical coordinates).
        f_original = f

        def f_transformed(x, y):
            x_phys, y_phys = transform_to_physical(x, y)
            return f_original(x_phys, y_phys)

        f = f_transformed

        if verbose:
            print(f"Domain transform: [0,1]² → [{domain['x1']},{domain['x2']}]×[{domain['y1']},{domain['y2']}]" )
            print(f"Laplacian factors in grid coords: ax=1/scale_x^2={ax:.3e}, ay=1/scale_y^2={ay:.3e}")
    
    if n > 5:
        print("Error: degree n too large (max 5)")
        return _early_return(None, None, None, np.nan, np.nan)
    
    # Load or compute collocation data
    if CD is None:
        if verbose:
            print(f"Computing collocation data for degree {n}...")
        CD = compute_collocation_data(n, J_MAX=16)
    
    E_ext = CD['E']
    dim_E = E_ext.shape[1]
    bw = CD['bw']
    
    # -------------------------------------------------------------------------
    # Classification (following MATLAB collocation_2d.m exactly)
    # -------------------------------------------------------------------------
    if verbose:
        print("Classification...")
    t_start = time.time()
    
    # Create grid: t goes from -(n-1)/2-bw to H+(n-1)/2+bw
    # MATLAB: t = (-(n-1)/2-bw:H+(n-1)/2+bw)/H
    t_start_val = -(n - 1) / 2 - bw
    t_end_val = H + (n - 1) / 2 + bw
    t_range = np.arange(t_start_val, t_end_val + 1, dtype=NP_DTYPE)
    t = t_range / H
    
    x_grid, y_grid = np.meshgrid(t, t, indexing='ij')
    
    # Evaluate weight function and derivatives on extended grid
    w, wx, wy, wxx, wyy = wfct(x_grid, y_grid)
    
    # Collocation point indices: MATLAB uses 1+bw:H+n+bw (1-based)
    # In Python (0-based): bw:H+n+bw, which gives H+n elements
    indB_start = bw
    indB_end = H + n + bw  # exclusive end, so H+n elements
    indB = np.arange(indB_start, indB_end)
    
    # Extract B-spline center grids
    xB = x_grid[indB, :][:, indB]  # Shape: (H+n, H+n)
    yB = y_grid[indB, :][:, indB]
    wB = w[indB, :][:, indB]
    
    dim_B = H + n  # Number of B-splines in each direction
    
    # Inner B-splines: those with centers inside domain (wB > 0)
    Btype_inner = wB > 0
    IL = np.where(Btype_inner.flatten())[0]  # Linear indices of inner B-splines
    mi = len(IL)
    
    if mi == 0:
        print("Error: No inner B-splines found. Grid may be too coarse.")
        return _early_return(None, None, None, np.nan, 0)
    
    # Create index mapping: Bindex[k1, k2] = reduced index for inner B-spline (k1, k2)
    # MATLAB: Bindex(IL)=1:mi (1-based), we use 0-based
    Bindex = np.zeros((dim_B, dim_B), dtype=np.int64) - 1  # -1 means not inner
    Bindex.flat[IL] = np.arange(mi)
    
    # Find arrays of (n+1)x(n+1) inner B-splines (for extension reference points)
    # MATLAB: eI = true(H,H); then eI = eI & Btype(s1:s1+H-1,s2:s2+H-1) for s1,s2 in 1:n+1
    # This checks if all B-splines in an (n+1)x(n+1) block are inner
    eI = np.ones((H, H), dtype=bool)
    for s1 in range(n + 1):
        for s2 in range(n + 1):
            # Check if Btype_inner at offset (s1, s2) is true
            eI = eI & Btype_inner[s1:s1 + H, s2:s2 + H]
    
    if not np.any(eI):
        print("Error: Grid width 1/H too large - no fully inner element arrays")
        return _early_return(None, None, None, np.nan, 0)
    
    # Centers of inner element arrays
    # MATLAB: (1/2:H)/H gives centers at 0.5/H, 1.5/H, ..., (H-0.5)/H
    xe_1d = (np.arange(H) + 0.5) / H
    ye_1d = (np.arange(H) + 0.5) / H
    xe_grid, ye_grid = np.meshgrid(xe_1d, ye_1d, indexing='ij')
    xe = xe_grid[eI]
    ye = ye_grid[eI]
    
    # Outer B-splines: active but center outside domain
    # MATLAB: for k1=-bw:bw, for k2=-bw:bw: Btype=Btype | w(k1+indB,k2+indB)>0
    Btype_active = Btype_inner.copy()
    for k1 in range(-bw, bw + 1):
        for k2 in range(-bw, bw + 1):
            # Get weight at offset positions
            w_idx1 = indB + k1
            w_idx2 = indB + k2
            # Ensure indices are valid
            valid1 = (w_idx1 >= 0) & (w_idx1 < w.shape[0])
            valid2 = (w_idx2 >= 0) & (w_idx2 < w.shape[1])
            if np.all(valid1) and np.all(valid2):
                w_offset = w[w_idx1, :][:, w_idx2]
                Btype_active = Btype_active | (w_offset > 0)
    
    JL = np.where((Btype_active.flatten()) & (~Btype_inner.flatten()))[0]
    mj = len(JL)
    
    # Get (j1, j2) coordinates of outer B-splines
    J1, J2 = np.unravel_index(JL, (dim_B, dim_B))
    
    rtimes['sys'] = time.time() - t_start
    
    if verbose:
        print(f"  Inner B-splines: {mi}")
        print(f"  Outer B-splines: {mj}")
        print(f"  Inner element arrays: {np.sum(eI)}")
    
    # -------------------------------------------------------------------------
    # Extension Matrix T (MATLAB: T(partind(:),m) = CD.E(:,D1(1),D1(2),D0(1),D0(2)))
    # T is (mi, mj): T[inner_idx, outer_idx] = extension coefficient
    # -------------------------------------------------------------------------
    if verbose:
        print("Building extension matrix...")
    t_start = time.time()
    
    T_rows = []
    T_cols = []
    T_data = []
    
    for m in range(mj):
        j1 = J1[m]  # 0-based index of outer B-spline
        j2 = J2[m]
        
        # Find closest inner element array center to outer B-spline center
        xB_j = xB[j1, j2]
        yB_j = yB[j1, j2]
        
        d_sq = (xe - xB_j) ** 2 + (ye - yB_j) ** 2
        ind_closest = np.argmin(d_sq)
        xej = xe[ind_closest]
        yej = ye[ind_closest]
        
        # Index of array corner in B-spline grid (MATLAB: IJ = round([xej,yej]*H+1/2))
        # This gives 1-based index in MATLAB. In Python 0-based: round(x*H + 0.5) - 1
        IJ = np.round(np.array([xej, yej]) * H + 0.5).astype(int) - 1  # 0-based
        
        # Distance from outer B-spline to array (MATLAB: D = [j1,j2]-IJ-n/2)
        # Note: MATLAB j1,j2 are 1-based, our j1,j2 are 0-based
        # MATLAB: D = [j1,j2]-IJ-n/2 where j1,j2 are 1-based, IJ is 1-based
        # Python: we need D = [j1+1, j2+1] - [IJ[0]+1, IJ[1]+1] - n/2 = [j1-IJ[0], j2-IJ[1]] - n/2
        D = np.array([j1, j2], dtype=float) - IJ - n / 2
        
        # D0: sign indicator (MATLAB: 1+(D>=0), gives 1 or 2 -> Python 0 or 1)
        D0 = ((D >= 0).astype(int))
        
        # D1: distance index (MATLAB: 1+fix(abs(D)), gives 1-based -> Python 0-based)
        D1 = np.floor(np.abs(D)).astype(int)
        
        if np.max(D1) >= dim_E:
            if verbose:
                print(f"Warning: Distance of outer B-spline {m} too large (D1={D1}, max={dim_E-1})")
            continue
        
        # Get extension coefficients
        E_coeffs = E_ext[:, D1[0], D1[1], D0[0], D0[1]]
        
        # Map to inner B-splines in the (n+1)x(n+1) array starting at IJ
        # MATLAB: partind = Bindex(IJ(1):IJ(1)+n, IJ(2):IJ(2)+n)
        for i1_local in range(n + 1):
            for i2_local in range(n + 1):
                inner_k1 = IJ[0] + i1_local
                inner_k2 = IJ[1] + i2_local
                
                if 0 <= inner_k1 < dim_B and 0 <= inner_k2 < dim_B:
                    inner_idx = Bindex[inner_k1, inner_k2]
                    if inner_idx >= 0:  # Valid inner B-spline
                        coeff = E_coeffs[i1_local * (n + 1) + i2_local]
                        if coeff != 0:
                            T_rows.append(inner_idx)
                            T_cols.append(m)
                            T_data.append(coeff)
    
    T = sparse.csr_matrix((T_data, (T_rows, T_cols)), shape=(mi, mj))
    
    rtimes['ext'] = time.time() - t_start
    
    if verbose:
        print(f"  Extension matrix: {mi} x {mj}, nnz = {T.nnz}")
    
    # -------------------------------------------------------------------------
    # Assemble Collocation Matrix (MATLAB: G(:,:,1+bw-s1,1+bw-s2) = ...)
    # -------------------------------------------------------------------------
    if verbose:
        print("Assembling collocation matrix...")
    t_start = time.time()
    
    G = np.zeros((dim_B, dim_B, 2 * bw + 1, 2 * bw + 1), dtype=NP_DTYPE)
    
    # Extract weight derivatives at collocation points
    wB_vals = wB  # Already extracted
    wxxB = wxx[indB, :][:, indB]
    wyyB = wyy[indB, :][:, indB]
    wxB = wx[indB, :][:, indB]
    wyB = wy[indB, :][:, indB]
    
    for s1 in range(-bw, bw + 1):
        for s2 in range(-bw, bw + 1):
            # B-spline indices (MATLAB: S1=1+bw+s1; Python: 0-indexed)
            S1 = bw + s1
            S2 = bw + s2
            
            # G array indices (MATLAB: 1+bw-s1; Python: 0-indexed)
            G_idx1 = bw - s1
            G_idx2 = bw - s2
            
            # B-spline values
            b00 = CD['b00'][S1, S2]
            b10 = CD['b10'][S1, S2]
            b01 = CD['b01'][S1, S2]
            b20 = CD['b20'][S1, S2]
            b02 = CD['b02'][S1, S2]
            
            # Physical Laplacian under affine mapping:
            #   -Δ_phys = -ax * ∂²_xgrid - ay * ∂²_ygrid,  ax=1/scale_x^2, ay=1/scale_y^2.
            # Applied to (w*B) and expanded:
            #   -Δ_phys(wB) = -ax*(w*B_xx + 2*w_x*B_x + B*w_xx) - ay*(w*B_yy + 2*w_y*B_y + B*w_yy)
            G[:, :, G_idx1, G_idx2] = (
                -H * H * ((ax * b20) + (ay * b02)) * wB_vals
                - b00 * ((ax * wxxB) + (ay * wyyB))
                - 2 * H * ((ax * b10) * wxB + (ay * b01) * wyB)
            )
    
    # -------------------------------------------------------------------------
    # Right-hand side
    # For non-homogeneous Dirichlet blending, modify RHS by Δ((1-w)g) - #! THe 1-w in not used
    # -------------------------------------------------------------------------
    F_rhs = f(xB, yB)
    # transform to physical coordinates if domain is specified
    if domain is not None:
        xB_phys, yB_phys = transform_to_physical(xB, yB)
    else:
        xB_phys, yB_phys = xB, yB
    points = torch.tensor(np.column_stack((xB_phys.flatten(), yB_phys.flatten())), dtype=torch.float64)
    #gB = ihbnd.get_lifting(model=model,function_case = function_case, num_sides=5, points = points).detach().cpu().numpy().reshape(xB.shape)

    if bundary_enforcement == "lifting":
        gB, gB_lp = ihbnd.get_lifting_laplacian(model=model,function_case = function_case, num_sides=5, points = points)
        gB = gB.detach().cpu().numpy().reshape(xB.shape) 
        gB_lp = gB_lp.detach().cpu().numpy().reshape(xB.shape)
    elif bundary_enforcement == "lightning":
        if function_case != 11:
            raise ValueError("bundary_enforcement='lightning' is currently only supported for function_case=11.")
        import lightning_laplace
        sol = lightning_laplace.solve_case_11(tolerance = 1e-6)
        gB = sol(xB_phys, yB_phys).reshape(xB.shape)
        gB_lp = np.zeros_like(xB)  # Set gB_lp to zero for now, as a placeholder #!todo
    elif bundary_enforcement == "none":
        gB = np.zeros_like(xB)  # Set gB to zero for now, as a placeholder #!todo
        gB_lp = np.zeros_like(xB)  # Set gB_lp to zero for now, as a placeholder #!todo
    elif bundary_enforcement == "wachspress":
        gB, gB_lp = ihbnd.get_lifting_polygon(model=model,function_case = function_case, num_sides=5, points = points,
                                              smoothness=1, blend_method="wachspress", eps=1e-12, return_laplacian=True)
        gB = gB.detach().cpu().numpy().reshape(xB.shape) 
        gB_lp = gB_lp.detach().cpu().numpy().reshape(xB.shape)
    elif bundary_enforcement == "coons":
        gB, gB_lp = ihbnd.get_lifting_polygon(model=model,function_case = function_case, num_sides=5, points = points,
                                            smoothness=1, blend_method="coons", eps=1e-12, return_laplacian=True)
        gB = gB.detach().cpu().numpy().reshape(xB.shape) 
        gB_lp = gB_lp.detach().cpu().numpy().reshape(xB.shape)
    else: 
        raise ValueError(f"Unknown boundary enforcement method: {bundary_enforcement}")

    # Δ_phys((1-w)g) in grid derivatives:
    #   ax * (-(wxx)g - 2 wx gx + (1-w) gxx) + ay * (-(wyy)g - 2 wy gy + (1-w) gyy)
    """F_rhs = F_rhs + (
        -((ax * wxxB) + (ay * wyyB)) * gB
    )"""
    #! Has been altered:
    F_rhs = F_rhs + gB_lp
    # Convert G array to sparse matrix
    SG = array_to_matrix(G)
    
    # Extract and apply extension (MATLAB: SG=SG(IL,IL)+SG(IL,JL)*T')
    SG_II = SG[np.ix_(IL, IL)]
    SG_IJ = SG[np.ix_(IL, JL)]
    SG_reduced = SG_II + SG_IJ @ T.T
    
    rtimes['sys'] += time.time() - t_start
    
    # -------------------------------------------------------------------------
    # Solve Linear System
    # -------------------------------------------------------------------------
    if verbose:
        print("Solving linear system...")
    t_start = time.time()
    
    F_inner = F_rhs.flat[IL]

    dim_sys = mi
    use_dense = (solve_method == "dense") or (solve_method == "auto" and dim_sys <= dense_threshold)
    if use_dense:
        if sparse.issparse(SG_reduced):
            SG_dense = SG_reduced.toarray()
        else:
            SG_dense = np.asarray(SG_reduced)
        SU = dense_solve(SG_dense, F_inner)
    else:
        if not sparse.issparse(SG_reduced):
            SG_reduced = sparse.csc_matrix(SG_reduced)
        try:
            SU = spsolve(SG_reduced.tocsc(), F_inner)
        except Exception as exc:
            raise RuntimeError(f"Sparse linear solve failed (mi={dim_sys}, method={solve_method}).") from exc
    
    # Extend coefficients to outer B-splines (MATLAB: U(JL)=T'*SU)
    U = np.zeros((dim_B, dim_B), dtype=NP_DTYPE)
    U.flat[IL] = SU
    if mj > 0:
        U.flat[JL] = (T.T @ SU)
    
    rtimes['sol'] = time.time() - t_start
    rtimes['total'] = rtimes['sys'] + rtimes['ext'] + rtimes['sol']
    
    # Condition number (only feasible for small dense systems)
    if dim_sys < 1000 and use_dense:
        if verbose:
            print("Estimating condition number...")
        try:
            con = np.linalg.cond(SG_dense)
        except:
            con = np.nan
    else:
        con = np.nan
    
    if verbose:
        print(f"  System dimension: {dim_sys}")
        if not np.isnan(con):
            print(f"  Condition number: {con:.2e}")
        else:
            print("  Condition number: N/A (system too large)")
    
    # -------------------------------------------------------------------------
    # Evaluate Solution (MATLAB: Uxy(ind1,ind2) += U.*max(w(ind1,ind2),0)*b00(s1,s2))
    # -------------------------------------------------------------------------
    if verbose:
        print("Evaluating solution...")
    
    b00 = CD['b00']
    dim_Uxy = dim_B + 2 * bw
    Uxy = np.zeros((dim_Uxy, dim_Uxy), dtype=NP_DTYPE)
    
    # MATLAB: for s1=1:1+2*bw, ind1=(s1:s1+H+n-1), so Python s1=0:2*bw+1, ind1=s1:s1+H+n
    for s1 in range(2 * bw + 1):
        for s2 in range(2 * bw + 1):
            # Output indices in Uxy
            ind1 = slice(s1, s1 + dim_B)
            ind2 = slice(s2, s2 + dim_B)
            
            # Weight values: need w at the correct grid positions
            # In extended grid, the positions for s1,s2 start at different offsets
            # MATLAB uses w(ind1, ind2) where ind1/ind2 go from s1 to s1+H+n-1
            # The extended grid w has size len(t) x len(t)
            # w[s1:s1+dim_B, s2:s2+dim_B] corresponds to the right positions
            w_slice = np.maximum(w[s1:s1 + dim_B, s2:s2 + dim_B], 0)
            
            Uxy[ind1, ind2] += U * w_slice * b00[s1, s2]
    
    # Extract result at collocation points (MATLAB: Uxy=Uxy(indB,indB))
    # In Python, we need to map back to the correct indices
    # The final Uxy should match xB, yB which are at indices indB-indB[0] in Uxy
    Uxy_result = Uxy[bw:bw + dim_B, bw:bw + dim_B]

    # Add the prescribed boundary extension part: (1-w)g
    Uxy_total = Uxy_result + 0 * (1.0 - wB_vals) * gB + gB #! TODO: Add boundary extension if needed
    
    if return_coefficients:
        recon_info = {
            'U': U.copy(),
            'n': n,
            'H': H,
            'dim_B': dim_B,
            'bw': bw,
            'domain': domain,
            'function_case': function_case,
        }
        return Uxy_total, xB, yB, con, dim_sys, rtimes, recon_info

    return Uxy_total, xB, yB, con, dim_sys, rtimes


# =============================================================================
# ARBITRARY-POINT B-SPLINE RECONSTRUCTION
# =============================================================================

def reconstruct_collocation_at_points(
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    recon_info: Dict,
    wfct_phys: 'NeuralWeightFunction',
    model_ms: Optional[torch.nn.Module] = None,
    lifting_method: Optional[str] = "get_lifting",
    function_case: Optional[int] = None,
) -> np.ndarray:
    """Evaluate collocation solution at arbitrary physical-coordinate points.

    Uses the B-spline basis expansion:
        v(x_g, y_g) = sum_{j1,j2} U[j1,j2] * B_n(tau_x - j1) * B_n(tau_y - j2)
        u = max(w, 0) * v

    where tau_x = x_g * H + n, tau_y = y_g * H + n, and x_g is the grid
    coordinate in [0, 1].

    Parameters
    ----------
    pts_x, pts_y : array-like
        Physical coordinates of evaluation points.
    recon_info : dict
        Reconstruction data returned by collocation_2d(return_coefficients=True).
    wfct_phys : NeuralWeightFunction
        Weight function that evaluates w in physical coordinates (domain=None).

    Returns
    -------
    u_h : ndarray
        Solution values at the given points.
    """
    pts_x = np.asarray(pts_x, dtype=NP_DTYPE).ravel()
    pts_y = np.asarray(pts_y, dtype=NP_DTYPE).ravel()

    U = recon_info['U']
    n = recon_info['n']
    H = recon_info['H']
    dim_B = recon_info['dim_B']
    domain = recon_info['domain']
    if function_case is None:
        function_case = int(recon_info.get('function_case', PDE_testcases.FUNCTION_CASE))

    # Map physical coords to grid coords [0,1]
    if domain is not None:
        x_g = (pts_x - domain['x1']) / (domain['x2'] - domain['x1'])
        y_g = (pts_y - domain['y1']) / (domain['y2'] - domain['y1'])
    else:
        x_g = pts_x.copy()
        y_g = pts_y.copy()

    # B-spline parameter: tau = x_g * H + n
    tau_x = x_g * H + n
    tau_y = y_g * H + n

    # Evaluate B_n(tau - j) for j = 0..dim_B-1
    # Shape: (N_pts, dim_B)
    j_vals = np.arange(dim_B, dtype=NP_DTYPE)
    Bx = bspline_evaluate(tau_x[:, None] - j_vals[None, :], n)  # (N, dim_B)
    By = bspline_evaluate(tau_y[:, None] - j_vals[None, :], n)  # (N, dim_B)

    # v = sum_{j1,j2} U[j1,j2] * Bx[:,j1] * By[:,j2]
    # Efficient: v = diag(Bx @ U @ By^T) = sum over j2 of (Bx @ U)[:,j2] * By[:,j2]
    v = np.sum((Bx @ U) * By, axis=1)

    # Weight in physical coords
    w_vals, *_ = wfct_phys(pts_x, pts_y)
    w_vals = np.asarray(w_vals, dtype=NP_DTYPE).ravel()

    u_h = np.maximum(w_vals, 0.0) * v 

    if model_ms is not None:
        lift_points = torch.tensor(np.column_stack((pts_x, pts_y)), dtype=torch.float64)
        if lifting_method == "get_lifting":
            lifting = ihbnd.get_lifting(model=model_ms,function_case = function_case, num_sides=5, points = lift_points).detach().cpu().numpy()
            u_h += lifting
        elif lifting_method == "lightning":
            if function_case != 11:
                raise ValueError("lifting_method='lightning' is currently only supported for function_case=11.")
            import lightning_laplace
            sol = lightning_laplace.solve_case_11(tolerance = 1e-6)
            lifting = sol(pts_x, pts_y)
            u_h += lifting
        elif lifting_method == "none":
            pass  # No lifting applied
        elif lifting_method == "wachspress":
            lifting = ihbnd.get_lifting_polygon(model=model_ms,function_case = function_case, num_sides=5, points = lift_points,
                                              smoothness=1, blend_method="wachspress", eps=1e-12, return_laplacian=False)
            lifting = lifting.detach().cpu().numpy()
            u_h += lifting
        elif lifting_method == "coons":
            lifting = ihbnd.get_lifting_polygon(model=model_ms,function_case = function_case, num_sides=5, points = lift_points,
                                            smoothness=1, blend_method="coons", eps=1e-12, return_laplacian=False)
            lifting = lifting.detach().cpu().numpy()
            u_h += lifting
        else:
            raise ValueError(f"Unknown lifting method: {lifting_method}")
    else: print("AAAAAAA")

    return u_h


def reconstruct_collocation_hessian_diag(
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    recon_info: Dict,
    wfct_phys: 'NeuralWeightFunction',
) -> tuple:
    """Evaluate collocation solution value, gradient, and Hessian diagonal
    at arbitrary physical-coordinate points.

    The collocation solution is u = w * v (homogeneous Dirichlet), so:
        u_x  = w_x * v + w * v_x
        u_xx = w_xx * v + 2 * w_x * v_x + w * v_xx
    (analogous for y).

    Parameters
    ----------
    pts_x, pts_y : array-like
        Physical coordinates of evaluation points.
    recon_info : dict
        Reconstruction data returned by collocation_2d(return_coefficients=True).
    wfct_phys : NeuralWeightFunction
        Weight function that evaluates (w, wx, wy, wxx, wyy) in physical coords.

    Returns
    -------
    u, u_x, u_y, u_xx, u_yy : ndarray
        Solution value, first and second partial derivatives.
    """
    pts_x = np.asarray(pts_x, dtype=NP_DTYPE).ravel()
    pts_y = np.asarray(pts_y, dtype=NP_DTYPE).ravel()

    U = recon_info['U']
    n = recon_info['n']
    H = recon_info['H']
    dim_B = recon_info['dim_B']
    domain = recon_info['domain']

    # Map physical coords to grid coords [0,1]
    if domain is not None:
        scale_x = domain['x2'] - domain['x1']
        scale_y = domain['y2'] - domain['y1']
        x_g = (pts_x - domain['x1']) / scale_x
        y_g = (pts_y - domain['y1']) / scale_y
    else:
        scale_x = 1.0
        scale_y = 1.0
        x_g = pts_x.copy()
        y_g = pts_y.copy()

    # B-spline parameters
    tau_x = x_g * H + n
    tau_y = y_g * H + n

    j_vals = np.arange(dim_B, dtype=NP_DTYPE)
    arg_x = tau_x[:, None] - j_vals[None, :]  # (N, dim_B)
    arg_y = tau_y[:, None] - j_vals[None, :]  # (N, dim_B)

    # B-spline values and derivatives w.r.t. tau
    Bx = bspline_evaluate(arg_x, n)
    By = bspline_evaluate(arg_y, n)
    dBx = bspline_derivative(arg_x, n, order=1)
    dBy = bspline_derivative(arg_y, n, order=1)
    d2Bx = bspline_derivative(arg_x, n, order=2)
    d2By = bspline_derivative(arg_y, n, order=2)

    # Chain rule: d/dx_phys = (H / scale_x) * d/dtau
    hsx = H / scale_x
    hsy = H / scale_y

    # Compute v and its physical-coordinate derivatives
    BxU = Bx @ U   # (N, dim_B)
    v = np.sum(BxU * By, axis=1)
    v_x = hsx * np.sum((dBx @ U) * By, axis=1)
    v_y = hsy * np.sum(BxU * dBy, axis=1)
    v_xx = (hsx ** 2) * np.sum((d2Bx @ U) * By, axis=1)
    v_yy = (hsy ** 2) * np.sum(BxU * d2By, axis=1)

    # Weight function in physical coords
    w_vals, wx, wy, wxx, wyy = wfct_phys(pts_x, pts_y)
    w_vals = np.asarray(w_vals, dtype=NP_DTYPE).ravel()
    wx = np.asarray(wx, dtype=NP_DTYPE).ravel()
    wy = np.asarray(wy, dtype=NP_DTYPE).ravel()
    wxx = np.asarray(wxx, dtype=NP_DTYPE).ravel()
    wyy = np.asarray(wyy, dtype=NP_DTYPE).ravel()

    # Clip w for outside points
    w_pos = np.maximum(w_vals, 0.0)

    # u = w * v  (using max(w,0) to enforce 0 outside)
    u = w_pos * v

    # Derivatives (product rule, only meaningful where w > 0)
    u_x = wx * v + w_vals * v_x
    u_y = wy * v + w_vals * v_y
    u_xx = wxx * v + 2.0 * wx * v_x + w_vals * v_xx
    u_yy = wyy * v + 2.0 * wy * v_y + w_vals * v_yy

    # Zero out outside domain
    outside = w_vals <= 0
    u[outside] = 0.0
    u_x[outside] = 0.0
    u_y[outside] = 0.0
    u_xx[outside] = 0.0
    u_yy[outside] = 0.0

    return u, u_x, u_y, u_xx, u_yy


# =============================================================================
# EXAMPLE: EXACT SOLUTION AND RHS
# =============================================================================

def example_exact_solution_disc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Exact solution for the disc domain.
    u_exact(x,y) = exp(w(x,y)) - 1, where w is the disc weight function.
    """
    w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
    return np.exp(w) - 1


def example_rhs_disc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side f = -Δu_exact for the disc domain.
    """
    # Derivatives of w
    wx = -4 * (2 * x - 1)
    wy = -4 * (2 * y - 1)
    wxx = -8.0
    wyy = -8.0
    w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
    
    # f = -exp(w) * (|∇w|^2 + Δw)
    f = -np.exp(w) * (wx ** 2 + wy ** 2 + wxx + wyy)
    return f


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

def compute_numerical_gradient(U: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute numerical gradient of solution array using central differences.
    
    Parameters:
    -----------
    U : ndarray
        2D solution array
    h : float
        Grid spacing (1/H)
    
    Returns:
    --------
    dU_dx, dU_dy : ndarray
        Gradient components
    """
    # Central differences for interior, forward/backward at boundaries
    dU_dx = np.zeros_like(U)
    dU_dy = np.zeros_like(U)
    
    # Central differences for interior
    dU_dx[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2 * h)
    dU_dy[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * h)
    
    # Forward/backward at boundaries
    dU_dx[0, :] = (U[1, :] - U[0, :]) / h
    dU_dx[-1, :] = (U[-1, :] - U[-2, :]) / h
    dU_dy[:, 0] = (U[:, 1] - U[:, 0]) / h
    dU_dy[:, -1] = (U[:, -1] - U[:, -2]) / h
    
    return dU_dx, dU_dy


def run_example(
    model,
    n: int = 2,
    H: int = 20,
    function_case: Optional[int] = None,
    verbose: bool = True,
    show_plot: bool = True,
    CD: Optional[Dict] = None,
    solve_method: str = "auto",
    dense_threshold: int = 2000,
    plot_mode: str = "3d",
    return_coefficients: bool = False,
    model_ms = None,
    boundary_enforcement: str = "lifting"
):
    """
    Run the collocation example.
    
    Parameters:
    -----------
    n : int
        B-spline degree (2-5)
    H : int
        Grid resolution (grid width = 1/H)
    domain : str
        Domain type: 'disc' or 'shovel'
    function_case : int, optional
        Function case (0-7). If None, uses global FUNCTION_CASE.
        0 = default WEB-spline example (exp(w)-1)
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary with solution, errors, and timing information
    """
    DOMAIN = PDE_testcases.DOMAIN
    old_case = PDE_testcases.FUNCTION_CASE
    old_domain = PDE_testcases.DOMAIN
    old_domain_local = DOMAIN
    if function_case is not None:
        PDE_testcases.set_function_case(function_case)
    active_case = PDE_testcases.FUNCTION_CASE
    DOMAIN = PDE_testcases.DOMAIN

    try:
        if verbose:
            print("=" * 60)
            print(f"WEB-Spline Collocation: degree={n}, H={H}, domain={DOMAIN}, case={active_case}")
            print("=" * 60)

        wfct = NeuralWeightFunction(model=model, domain=DOMAIN)
        transformer = create_domain_transformer(DOMAIN)

        if CD is None:
            CD = compute_collocation_data(n, J_MAX=16)

        collocation_result = collocation_2d(
            n,
            H,
            wfct,
            f=PDE_testcases.load_function_vectorized,
            CD=CD,
            verbose=verbose,
            domain=DOMAIN,
            solve_method=solve_method,
            dense_threshold=dense_threshold,
            model=model_ms,
            function_case=active_case,
            return_coefficients=return_coefficients,
            bundary_enforcement=boundary_enforcement
        )

        if return_coefficients:
            Uxy, xB, yB, con, dim_sys, rtimes, recon_info = collocation_result
        else:
            Uxy, xB, yB, con, dim_sys, rtimes = collocation_result
            recon_info = None

        if Uxy is None:
            print("Solver failed!")
            raise RuntimeError("Collocation solver failed.")

        w_grid, _, _, _, _ = wfct(xB, yB)
        mask = w_grid > 0

        u_exact_transformed = transformer.wrap_function(PDE_testcases.solution_function)
        du_dx_transformed = transformer.wrap_derivative_x(PDE_testcases.solution_function_derivative_x)
        du_dy_transformed = transformer.wrap_derivative_y(PDE_testcases.solution_function_derivative_y)

        uxy_exact = np.zeros_like(Uxy)
        uxy_exact[mask] = u_exact_transformed(xB[mask], yB[mask])

        error = Uxy[mask] - uxy_exact[mask]
        exact_vals = uxy_exact[mask]

        max_exact = np.max(np.abs(exact_vals)) if np.any(exact_vals != 0) else 1.0
        l2_exact = np.sqrt(np.sum(exact_vals ** 2)) if np.any(exact_vals != 0) else 1.0

        ErrMax = np.max(np.abs(error)) / max_exact
        ErrL2 = np.sqrt(np.sum(error ** 2)) / l2_exact
        Err_MAE = np.mean(np.abs(error))
        Err_L_inf = np.max(np.abs(error))

        h = 1.0 / H
        dU_dx_num, dU_dy_num = compute_numerical_gradient(Uxy, h)

        H1_semi = np.nan
        try:
            dU_dx_ex = np.zeros_like(Uxy)
            dU_dy_ex = np.zeros_like(Uxy)
            dU_dx_ex[mask] = du_dx_transformed(xB[mask], yB[mask])
            dU_dy_ex[mask] = du_dy_transformed(xB[mask], yB[mask])

            grad_error_x = dU_dx_num[mask] - dU_dx_ex[mask]
            grad_error_y = dU_dy_num[mask] - dU_dy_ex[mask]
            grad_exact_norm = np.sqrt(np.sum(dU_dx_ex[mask] ** 2 + dU_dy_ex[mask] ** 2))

            if grad_exact_norm > 0:
                H1_semi = np.sqrt(np.sum(grad_error_x ** 2 + grad_error_y ** 2)) / grad_exact_norm
        except NotImplementedError:
            H1_semi = np.nan

        H1_error = np.sqrt(ErrL2 ** 2 + H1_semi ** 2) if not np.isnan(H1_semi) else np.nan

        if verbose:
            print("\n" + "=" * 40)
            print("Results:")
            print("=" * 40)
            print(f"  Relative max error: {ErrMax:.6e}")
            print(f"  Relative L2 error:  {ErrL2:.6e}")
            print(f"  Mean absolute error: {Err_MAE:.6e}")
            print(f"  Absolute L-inf error: {Err_L_inf:.6e}")
            print(f"  H1 semi-norm error: {H1_semi:.6e}" if not np.isnan(H1_semi) else "  H1 semi-norm error: N/A")
            print(f"  H1 error:           {H1_error:.6e}" if not np.isnan(H1_error) else "  H1 error:           N/A")
            print(f"  Condition number:   {con:.2e}" if not np.isnan(con) else "  Condition number:   N/A")
            print(f"  System dimension:   {dim_sys}")
            print(f"\nTiming:")
            print(f"  Classification/Assembly: {rtimes['sys']:.3f}s")
            print(f"  Extension:               {rtimes['ext']:.3f}s")
            print(f"  Solution:                {rtimes['sol']:.3f}s")
            print(f"  Total:                   {rtimes['total']:.3f}s")

        result = {
            'Uxy': Uxy,
            'xB': xB,
            'yB': yB,
            'uxy_exact': uxy_exact,
            'ErrMax': ErrMax,
            'ErrL2': ErrL2,
            'MAE': Err_MAE,
            'H1_semi': H1_semi,
            'H1_error': H1_error,
            'condition': con,
            'dim': dim_sys,
            'rtimes': rtimes,
            'wfct': wfct,
            'function_case': active_case,
        }

        if recon_info is not None:
            result['recon_info'] = recon_info

        if show_plot:
            visualize_solution(result, plot_mode=plot_mode)

        return result
    finally:
        PDE_testcases.FUNCTION_CASE = old_case
        PDE_testcases.DOMAIN = old_domain
        DOMAIN = old_domain_local


def visualize_solution(results: Dict, show_error: bool = True, plot_mode: str = "3d"):
    """
    Visualize the solution and error.
    
    Parameters:
    -----------
    results : dict
        Output from run_example()
    show_error : bool
        Also plot the error distribution
    plot_mode : str
        Plotting style: '3d' (surface) or 'heatmap' (2d).
    """
    import matplotlib.pyplot as plt

    plot_mode_norm = (plot_mode or "3d").strip().lower()
    if plot_mode_norm in {"2d", "heatmap", "hm"}:
        plot_mode_norm = "heatmap"
    elif plot_mode_norm in {"3d", "surface", "surf"}:
        plot_mode_norm = "3d"
    else:
        raise ValueError(f"Unknown plot_mode='{plot_mode}'. Use '3d' or 'heatmap'.")
    
    Uxy = results['Uxy']
    xB = results['xB']
    yB = results['yB']
    uxy_exact = results['uxy_exact']
    wfct = results['wfct']
    
    w_grid, _, _, _, _ = wfct(xB, yB)
    mask = w_grid > 0

    # Only visualize values inside the physical domain.
    Uxy_plot = np.array(Uxy, copy=True)
    uxy_exact_plot = np.array(uxy_exact, copy=True)
    Uxy_plot[~mask] = np.nan
    uxy_exact_plot[~mask] = np.nan

    def _plot_heatmap(ax, Z, title: str, cmap: str):
        # pcolormesh works for general (xB,yB) grids.
        mappable = ax.pcolormesh(xB, yB, Z, shading='auto', cmap=cmap)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    
    error = np.abs(Uxy - uxy_exact)
    error[~mask] = np.nan

    if plot_mode_norm == "heatmap":
        if show_error:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
            _plot_heatmap(ax1, Uxy_plot, 'Numerical Solution', cmap='viridis')
            _plot_heatmap(ax2, uxy_exact_plot, 'Exact Solution', cmap='viridis')
            _plot_heatmap(ax3, error, f'Absolute Error (max={results["ErrMax"]:.2e})', cmap='hot')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6), constrained_layout=True)
            _plot_heatmap(ax, Uxy_plot, 'WEB-Spline Collocation Solution', cmap='viridis')
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if show_error:
            fig = plt.figure(figsize=(14, 5))

            # Numerical solution
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(xB, yB, Uxy_plot, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('u')
            ax1.set_title('Numerical Solution')

            # Exact solution
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot_surface(xB, yB, uxy_exact_plot, cmap='viridis', alpha=0.8)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('u')
            ax2.set_title('Exact Solution')

            # Error
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot_surface(xB, yB, error, cmap='hot', alpha=0.8)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('|error|')
            ax3.set_title(f'Absolute Error (max={results["ErrMax"]:.2e})')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xB, yB, Uxy_plot, cmap='viridis', alpha=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title('WEB-Spline Collocation Solution')
    
    plt.show()


def convergence_study(
    model,
    n_list: list = None,
    H_list: list = None,
    function_case: int = None,
):
    """
    Perform a convergence study.
    
    Parameters:
    -----------
    n_list : list
        List of B-spline degrees to test
    H_list : list
        List of grid resolutions to test
    domain : str
        Domain type
    function_case : int
        Test problem case (0-7). If None, uses global FUNCTION_CASE.
    
    Returns:
    --------
    results : dict
        Dictionary with errors and timings for each (n, H) pair
    """
    if n_list is None:
        n_list = [2, 3, 4]
    if H_list is None:
        H_list = [10, 20, 40, 80]
    
    results = {}
    
    fc = function_case if function_case is not None else FEM.FUNCTION_CASE
    
    print("=" * 110)
    print(f"Convergence Study (FUNCTION_CASE={fc}, domain={DOMAIN})")
    print("=" * 110)
    print(f"{'n':>3} | {'H':>4} | {'ErrMax':>10} | {'ErrL2':>10} | {'MAE':>10} | {'H1_semi':>10} | {'H1_err':>10} | {'Cond':>10} | {'Time':>8}")
    print("-" * 110)
    
    for n in n_list:
        CD = compute_collocation_data(n, J_MAX=16)
        
        for H in H_list:
            res = run_example(model, n=n, H=H, function_case=function_case, verbose=False, show_plot=False, CD=CD)
            if res is not None:
                key = (n, H)
                results[key] = res
                print(f"{n:>3} | {H:>4} | {res['ErrMax']:>10.2e} | {res['ErrL2']:>10.2e} | "
                      f"{res['MAE']:>10.2e} | {res['H1_semi']:>10.2e} | {res['H1_error']:>10.2e} | "
                      f"{res['condition']:>10.2e} | {res['rtimes']['total']:>7.3f}s")
            else:
                print(f"{n:>3} | {H:>4} | {'FAILED':>10} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8}")
    
    # Print convergence orders if enough data points
    print("-" * 110)
    for n in n_list:
        H_prev = None
        err_prev = None
        for H in H_list:
            key = (n, H)
            if key in results:
                if H_prev is not None and err_prev is not None:
                    # Compute convergence order from ErrL2
                    order = np.log(err_prev / results[key]['ErrL2']) / np.log(H / H_prev)
                    print(f"n={n}, H: {H_prev} -> {H}, L2 order: {order:.2f}")
                H_prev = H
                err_prev = results[key]['ErrL2']
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Run this module from a notebook/script. Example:")
    print("  from Geomertry import AnaliticalDistanceCircle")
    print("  import collocation_WEB as cWEB")
    print("  res = cWEB.run_example(AnaliticalDistanceCircle(), n=3, H=50, function_case=1)")

    import network_defs as netdefs
    model_MS = netdefs.load_test_model("SIREN_pentagon_MSSDF_large", "SIREN", params={"architecture": [2, 256, 256, 256, 5], "w_0": 15, "w_hidden": 30.0})
    model = ihbnd.sharp_min_distance_function(model_MS)
    res = run_example(model=model, n=4, H=100, function_case=11, verbose=True, plot_mode='2d', return_coefficients=True, model_ms=model_MS)
    result = ihbnd.plot_case_11_poisson_samples(
        recon_info=res['recon_info'],
        model=model,
        title_prefix="Case 11: Pentagon Domain",
        show=True,
        filename="poisson_samples.csv",
        model_ms=model_MS
    )


