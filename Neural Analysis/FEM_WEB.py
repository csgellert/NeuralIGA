"""
WEB-splines (Weighted Extended B-splines) implementation for Neural IGA.

Based on:
    Höllig, K., Reif, U., Wipper, J. (2001). "Weighted Extended B-Spline 
    Approximation of Dirichlet Problems." Mathematics of Computation, 70(235), 51-63.

Key differences from standard weighted B-splines:
    1. B-splines are classified as 'inner' (stable) or 'outer' (unstable)
    2. Outer B-splines are extended by expressing them as linear combinations of inner B-splines
    3. The final basis consists only of WEB-splines (weighted extended B-splines)
    4. This eliminates ill-conditioning from B-splines with small support intersection

Author: Neural IGA Research
Date: 2026-01-21
"""

from Geomertry import *
import numpy as np
import mesh
import torch
from tqdm import tqdm
from scipy import linalg
from scipy import sparse

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)
NP_DTYPE = np.float64
TORCH_DTYPE = torch.float64

# Import shared functions from FEM module
from FEM import (
    FUNCTION_CASE, MAX_SUBDIVISION, DOMAIN,
    _get_gauss_points, load_function_vectorized,
    dirichletBoundary_vectorized, dirichletBoundaryDerivativeX_vectorized,
    dirichletBoundaryDerivativeY_vectorized, solution_function,
    solution_function_derivative_x, solution_function_derivative_y,
    _get_bspline_objects
)

# Import exact Höllig extension coefficient computation
from hollig_exact_extension import computeExtensionCoefficientsHollig


# =============================================================================
# WEB-SPLINE CLASSIFICATION
# =============================================================================


def classifyBsplinesHollig(element_types, p, q, knotvector_x, knotvector_y, xDivision, yDivision):
    """Höllig-style classification (element-driven).

    The original weighted B-spline solver classifies *elements* as inner/boundary/outer.
    In Höllig's WEB construction, a B-spline is:
      - inner if its support contains at least one *inner element* (cell fully in Ω)
      - outer if it has non-empty intersection with Ω but contains no full inner cell
        (in practice: active on at least one *boundary element* but not inner)
      - completely outside otherwise.

    This avoids heuristic “support fraction” thresholds and directly follows the
    cell-based definition.
    """
    n_basis_x = len(knotvector_x) - p - 1
    n_basis_y = len(knotvector_y) - q - 1

    inner_active = set()
    boundary_active = set()

    def _mark_active(elem_list, target_set):
        for elemx, elemy, *_ in elem_list:
            i_start = max(elemx - p, 0)
            i_end = min(elemx, n_basis_x - 1)
            j_start = max(elemy - q, 0)
            j_end = min(elemy, n_basis_y - 1)
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    target_set.add((i, j))

    _mark_active(element_types.get("inner", []), inner_active)
    _mark_active(element_types.get("boundary", []), boundary_active)

    inner_bsplines = sorted(inner_active)
    outer_bsplines = sorted(boundary_active.difference(inner_active))

    all_basis = {(i, j) for i in range(n_basis_x) for j in range(n_basis_y)}
    completely_outside = sorted(all_basis.difference(inner_active.union(boundary_active)))

    # Build mapping from inner B-splines to reduced indices
    inner_to_reduced_idx = {}
    reduced_to_global_idx = {}
    for reduced_idx, (i, j) in enumerate(inner_bsplines):
        global_idx = i * n_basis_y + j
        inner_to_reduced_idx[(i, j)] = reduced_idx
        reduced_to_global_idx[reduced_idx] = global_idx

    return {
        'inner': inner_bsplines,
        'outer': outer_bsplines,
        'completely_outside': completely_outside,
        'n_inner': len(inner_bsplines),
        'n_outer': len(outer_bsplines),
        'n_basis_x': n_basis_x,
        'n_basis_y': n_basis_y,
        'inner_to_reduced_idx': inner_to_reduced_idx,
        'reduced_to_global_idx': reduced_to_global_idx,
    }


def _compute_inner_reference_weights_from_inner_elements(
    model,
    bspline_classification,
    element_types,
    p,
    q,
    eps: float = 1e-10,
):
    """Compute per-inner-basis reference weights w(x_i) for WEB normalization.

    For each inner tensor-product B-spline index (i,j), we select x_i as the
    center of a fully-inside ("inner") element contained in its support.
    Then w(x_i) is evaluated via the SDF model.

    This matches the paper's intent (choose x_i in an inner cell of supp(b_i)),
    and provides the missing factor w(x)/w(x_i) used in Definition 2.
    """
    inner_bsplines = bspline_classification['inner']
    inner_set = set(inner_bsplines)
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']

    inner_elements = element_types.get('inner', [])
    if len(inner_elements) == 0:
        raise RuntimeError(
            "WEB normalization requires at least one fully-inside element, but none were found."
        )

    # Map each inner basis to the center of an inner element in its support.
    ref_points = {}
    for elemx, elemy, x1, x2, y1, y2 in inner_elements:
        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2)
        i_start = max(elemx - p, 0)
        i_end = min(elemx, n_basis_x - 1)
        j_start = max(elemy - q, 0)
        j_end = min(elemy, n_basis_y - 1)
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                idx = (i, j)
                if idx in inner_set and idx not in ref_points:
                    ref_points[idx] = (xc, yc)

    missing = [idx for idx in inner_bsplines if idx not in ref_points]
    if missing:
        raise RuntimeError(
            f"Could not assign a reference inner-element center x_i for {len(missing)} inner B-splines. "
            "This indicates an inconsistency between element classification and B-spline classification."
        )

    pts = np.asarray([ref_points[idx] for idx in inner_bsplines], dtype=NP_DTYPE)

    # Evaluate w(x_i) using vectorized mesh function (ignore derivatives)
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    w_tensor, _, _ = mesh.distance_with_derivative_vect_trasformed(
        x_coords, y_coords, model, transform=mesh.TRANSFORM
    )
    w = w_tensor.detach().cpu().numpy().astype(NP_DTYPE, copy=False)

    if np.any(~np.isfinite(w)):
        raise RuntimeError("Non-finite values encountered while evaluating w(x_i) for WEB normalization.")

    if np.any(w <= eps):
        bad = np.where(w <= eps)[0]
        min_w = float(np.min(w))
        raise RuntimeError(
            f"WEB normalization would divide by non-positive/small w(x_i). min w(x_i)={min_w:.3e}. "
            f"Count <= eps: {len(bad)} (eps={eps:.1e})."
        )

    ref_weights = {idx: float(val) for idx, val in zip(inner_bsplines, w)}
    return ref_weights, ref_points


# =============================================================================
# EXTENSION COEFFICIENTS COMPUTATION
# =============================================================================

# Legacy extension builders were removed. The solver only supports the
# paper-faithful Eq.(9) coefficient construction implemented in
# computeExtensionCoefficientsHollig (strict mode).


def buildExtendedBasis(bspline_classification, extension_coeffs):
    """
    Build the structure for extended B-splines.
    
    For each inner B-spline B_i, the extended B-spline is:
        B_i^e = B_i + Σ_{j ∈ outer} e_{ij} * B_j
    
    where e_{ij} are the extension coefficients (reversed indexing).
    
    Parameters:
    -----------
    bspline_classification : dict
        Output from classifyBsplines()
    extension_coeffs : dict
        Output from computeExtensionCoefficientsHollig()
    
    Returns:
    --------
    dict : Extended basis structure
        - Key: (inner_i, inner_j) - index of inner B-spline
        - Value: dict mapping (bspline_i, bspline_j) -> coefficient
                 Includes the inner B-spline itself with coefficient 1
    """
    inner_bsplines = bspline_classification['inner']
    inner_set = set(inner_bsplines)
    
    extended_basis = {}
    
    for inner_idx in inner_bsplines:
        # Start with the inner B-spline itself
        basis_coeffs = {inner_idx: 1.0}
        
        # Add contributions from outer B-splines that extend to this inner one
        for outer_idx, coeffs in extension_coeffs.items():
            if inner_idx in coeffs:
                # This outer B-spline extends to this inner B-spline
                # So we add the outer B-spline to the extended basis
                basis_coeffs[outer_idx] = coeffs[inner_idx]
        
        extended_basis[inner_idx] = basis_coeffs
    
    return extended_basis


# =============================================================================
# LOCAL DOF SELECTION + ASSEMBLY (WEB)
# =============================================================================

def _get_local_inner_dofs(elemx, elemy, p, q, bspline_classification, basis_to_inner):
    """Return the WEB-reduced inner DOFs that are nonzero on this element.

    NOTE: For WEB-splines, an extended basis function B_i^e can be nonzero on
    an element even if the underlying inner tensor-product B-spline b_i is not
    active there (because B_i^e mixes in outer splines). Therefore we determine
    local reduced DOFs via an inverse map:

        basis_to_inner[(k_x,k_y)] -> set of inner indices i such that
                                     (k_x,k_y) appears in B_i^e.

    For a tensor-product B-spline basis, the active (global) indices on the
    knot span [elemx, elemx+1]×[elemy, elemy+1] are:
      i in [elemx-p, elemx], j in [elemy-q, elemy]

    Returns:
        local_inner: list[(i,j)] inner indices active on this element (WEB sense)
        local_reduced: list[int] reduced indices corresponding to local_inner
    """
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']
    inner_to_reduced = bspline_classification['inner_to_reduced_idx']

    i_start = max(elemx - p, 0)
    i_end = min(elemx, n_basis_x - 1)
    j_start = max(elemy - q, 0)
    j_end = min(elemy, n_basis_y - 1)

    active_pairs = [(i, j) for i in range(i_start, i_end + 1) for j in range(j_start, j_end + 1)]
    local_inner_set = set()
    for pair in active_pairs:
        inners = basis_to_inner.get(pair)
        if inners:
            local_inner_set.update(inners)

    local_inner = sorted(local_inner_set)
    local_reduced = [inner_to_reduced[idx] for idx in local_inner]
    return local_inner, local_reduced


def _assembly_web(K, F, Ke, Fe, reduced_indices):
    """Assemble a local WEB element matrix/vector into the global reduced system (sparse version)."""
    if len(reduced_indices) == 0:
        return K, F
    idxs = np.asarray(reduced_indices, dtype=np.int64)
    # For sparse lil_matrix, element-wise addition is efficient
    for i_loc, i_glob in enumerate(idxs):
        for j_loc, j_glob in enumerate(idxs):
            K[i_glob, j_glob] += Ke[i_loc, j_loc]
    F[idxs] += Fe
    return K, F


# =============================================================================
# WEB-SPLINE QUADRATURE
# =============================================================================

def GaussQuadratureWEB(
    model,
    x1,
    x2,
    y1,
    y2,
    p,
    q,
    knotvector_x,
    knotvector_y,
    Bspxi,
    Bspeta,
    bspline_classification,
    extended_basis,
    local_inner,
):
    """
    Gauss quadrature for WEB-splines on a single element.
    
    Computes element stiffness matrix and load vector using extended B-splines.
    The result is a reduced system involving only inner B-spline DOFs.
    
    Parameters:
    -----------
    model : neural network
        SDF model for geometry
    x1, x2, y1, y2 : float
        Element bounds
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array
        Knot vectors
    Bspxi, Bspeta : Bspline objects
        B-spline basis evaluators
    bspline_classification : dict
        B-spline classification info
    extended_basis : dict
        Extended basis structure
    
    Returns:
    --------
    K_reduced : ndarray
        Element stiffness matrix (n_inner x n_inner)
    F_reduced : ndarray
        Element load vector (n_inner,)
    """
    gaussP_x, gaussP_y, gauss_weights, num_gauss_points = _get_gauss_points(p)
    
    n_inner = bspline_classification['n_inner']
    inner_bsplines = bspline_classification['inner']
    inner_to_reduced = bspline_classification['inner_to_reduced_idx']
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']
    
    # IMPORTANT: we only compute element contributions for DOFs that are
    # active on this element (local_inner), to avoid O(n_inner^2) work.
    n_loc = len(local_inner)
    K_reduced = np.zeros((n_loc, n_loc))
    F_reduced = np.zeros(n_loc)
    
    # Transform Gauss points to physical coordinates
    Jxi = (x2 - x1) / 2
    Jeta = (y2 - y1) / 2
    Jacobian = Jxi * Jeta
    
    xi = Jxi * gaussP_x + (x2 + x1) / 2
    eta = Jeta * gaussP_y + (y2 + y1) / 2
    
    # Get distance function and derivatives
    d_, dx_, dy_ = mesh.distance_with_derivative_vect_trasformed(xi, eta, model)
    
    if hasattr(d_, 'detach'):
        d_np = d_.detach().numpy().flatten()
        dx_np = dx_.detach().numpy().flatten()
        dy_np = dy_.detach().numpy().flatten()
    else:
        d_np = np.asarray(d_).flatten()
        dx_np = np.asarray(dx_).flatten()
        dy_np = np.asarray(dy_).flatten()
    
    # Valid Gauss points (inside domain)
    valid_mask = d_np >= 0
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return K_reduced, F_reduced
    
    d_valid = d_np[valid_mask]
    dx_valid = dx_np[valid_mask]
    dy_valid = dy_np[valid_mask]
    weights_valid = gauss_weights[valid_mask]
    xi_valid = xi[valid_mask]
    eta_valid = eta[valid_mask]
    
    # Evaluate all B-splines at Gauss points
    bxi_all = Bspxi.collmat(xi_valid)
    beta_all = Bspeta.collmat(eta_valid)
    dbdxi_all = Bspxi.collmat(xi_valid, 1)
    dbdeta_all = Bspeta.collmat(eta_valid, 1)
    
    if bxi_all.ndim == 1:
        bxi_all = bxi_all.reshape(1, -1)
    if beta_all.ndim == 1:
        beta_all = beta_all.reshape(1, -1)
    if dbdxi_all.ndim == 1:
        dbdxi_all = dbdxi_all.reshape(1, -1)
    if dbdeta_all.ndim == 1:
        dbdeta_all = dbdeta_all.reshape(1, -1)
    
    # Compute extended B-spline values and derivatives at each Gauss point
    # Extended B-spline: B_i^e = Σ_j e_ij * B_j
    
    # For each local inner B-spline (each local DOF)
    Be_values = np.zeros((n_valid, n_loc))      # Extended B-spline values
    dBe_xi_values = np.zeros((n_valid, n_loc))  # x-derivatives
    dBe_eta_values = np.zeros((n_valid, n_loc)) # y-derivatives

    for loc_col, inner_idx in enumerate(local_inner):
        # Get coefficients for this extended B-spline
        basis_coeffs = extended_basis[inner_idx]
        
        for (bi, bj), coeff in basis_coeffs.items():
            if bi < n_basis_x and bj < n_basis_y:
                # Tensor product: B(x,y) = B_xi(x) * B_eta(y)
                B_vals = bxi_all[:, bi] * beta_all[:, bj]
                dB_xi = dbdxi_all[:, bi] * beta_all[:, bj]
                dB_eta = bxi_all[:, bi] * dbdeta_all[:, bj]

                Be_values[:, loc_col] += coeff * B_vals
                dBe_xi_values[:, loc_col] += coeff * dB_xi
                dBe_eta_values[:, loc_col] += coeff * dB_eta

    # Optional WEB normalization (Definition 2): φ_i = (w / w(x_i)) * B_i^e
    # We implement this by scaling each column of B_i^e (and its derivatives)
    # by 1 / w(x_i), where x_i is chosen from a fully-inside element in supp(b_i).
    if bspline_classification.get('web_use_weight_normalization', False):
        ref_weights = bspline_classification.get('inner_reference_weights')
        if ref_weights is None:
            raise RuntimeError(
                "WEB normalization enabled but bspline_classification['inner_reference_weights'] is missing."
            )
        w_ref = np.asarray([ref_weights[idx] for idx in local_inner], dtype=NP_DTYPE)
        inv_w = (1.0 / w_ref)[np.newaxis, :]
        Be_values *= inv_w
        dBe_xi_values *= inv_w
        dBe_eta_values *= inv_w
    
    # Compute weighted gradients for WEB-splines
    # WEB-spline: φ_i = w * B_i^e, where w = d (distance function)
    # If normalization is enabled above, B_i^e already includes 1/w(x_i).
    # ∇φ_i = ∇w * B_i^e + w * ∇B_i^e
    grad_phi_xi = dx_valid[:, np.newaxis] * Be_values + d_valid[:, np.newaxis] * dBe_xi_values
    grad_phi_eta = dy_valid[:, np.newaxis] * Be_values + d_valid[:, np.newaxis] * dBe_eta_values
    
    # Load function and Dirichlet boundary values
    f_values = load_function_vectorized(xi_valid, eta_valid)
    dirichlet_values = dirichletBoundary_vectorized(xi_valid, eta_valid)
    dirichlet_dx = dirichletBoundaryDerivativeX_vectorized(xi_valid, eta_valid)
    dirichlet_dy = dirichletBoundaryDerivativeY_vectorized(xi_valid, eta_valid)
    
    if hasattr(f_values, 'flatten'):
        f_values = np.asarray(f_values).flatten()
        dirichlet_values = np.asarray(dirichlet_values).flatten()
        dirichlet_dx = np.asarray(dirichlet_dx).flatten()
        dirichlet_dy = np.asarray(dirichlet_dy).flatten()
    
    # Assemble K using einsum (bilinear form: ∫ ∇φ_i · ∇φ_j dx)
    weighted_grad_xi = grad_phi_xi * weights_valid[:, np.newaxis]
    weighted_grad_eta = grad_phi_eta * weights_valid[:, np.newaxis]
    
    K_reduced = (np.einsum('gi,gj->ij', weighted_grad_xi, grad_phi_xi) + 
                 np.einsum('gi,gj->ij', weighted_grad_eta, grad_phi_eta)) * Jacobian
    
    # Assemble F (load vector with Dirichlet lifting)
    # F = ∫ f * φ_i dx - ∫ ∇g · ∇φ_i dx  (where g is Dirichlet BC)
    phi_values = d_valid[:, np.newaxis] * Be_values  # WEB-spline values
    
    term1 = f_values[:, np.newaxis] * phi_values
    term2_xi = grad_phi_xi * (dx_valid * dirichlet_values + d_valid * dirichlet_dx)[:, np.newaxis]
    term2_eta = grad_phi_eta * (dy_valid * dirichlet_values + d_valid * dirichlet_dy)[:, np.newaxis]
    term3 = dirichlet_dx[:, np.newaxis] * grad_phi_xi + dirichlet_dy[:, np.newaxis] * grad_phi_eta
    
    F_contrib = (term1 + term2_xi + term2_eta - term3) * weights_valid[:, np.newaxis]
    F_reduced = np.sum(F_contrib, axis=0) * Jacobian
    
    return K_reduced, F_reduced


# =============================================================================
# SUBDIVISION FOR BOUNDARY ELEMENTS
# =============================================================================

def SubdivideWEB(model, x1, x2, y1, y2, p, q, knotvector_x, knotvector_y,
                 Bspxi, Bspeta, bspline_classification, extended_basis, local_inner,
                 level, MAXLEVEL=4):
    """
    Adaptive subdivision for WEB-spline integration on boundary elements.
    """
    n_loc = len(local_inner)
    
    if level == MAXLEVEL:
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        K = np.zeros((n_loc, n_loc))
        F = np.zeros(n_loc)
        
        for sub_x1, sub_x2, sub_y1, sub_y2 in [
            (x1, halfx, y1, halfy),
            (halfx, x2, y1, halfy),
            (x1, halfx, halfy, y2),
            (halfx, x2, halfy, y2)
        ]:
            Ksub, Fsub = GaussQuadratureWEB(
                model, sub_x1, sub_x2, sub_y1, sub_y2,
                p, q, knotvector_x, knotvector_y,
                Bspxi, Bspeta, bspline_classification, extended_basis,
                local_inner,
            )
            K += Ksub
            F += Fsub
        
        return K, F
    else:
        halfx = (x1 + x2) * 0.5
        halfy = (y1 + y2) * 0.5
        
        K = np.zeros((n_loc, n_loc))
        F = np.zeros(n_loc)
        
        for sub_x1, sub_x2, sub_y1, sub_y2 in [
            (x1, halfx, y1, halfy),
            (halfx, x2, y1, halfy),
            (x1, halfx, halfy, y2),
            (halfx, x2, halfy, y2)
        ]:
            Ksub, Fsub = SubdivideWEB(
                model, sub_x1, sub_x2, sub_y1, sub_y2,
                p, q, knotvector_x, knotvector_y,
                Bspxi, Bspeta, bspline_classification, extended_basis,
                local_inner,
                level + 1, MAXLEVEL,
            )
            K += Ksub
            F += Fsub
        
        return K, F


# =============================================================================
# ELEMENT PROCESSING
# =============================================================================

def elementWEB(model, p, q, knotvector_x, knotvector_y, elemx, elemy, x1, x2, y1, y2,
               Bspxi, Bspeta, bspline_classification, extended_basis, basis_to_inner):
    """Process inner element with WEB-splines (no subdivision)."""
    local_inner, local_reduced = _get_local_inner_dofs(elemx, elemy, p, q, bspline_classification, basis_to_inner)
    Ke, Fe = GaussQuadratureWEB(
        model, x1, x2, y1, y2, p, q, knotvector_x, knotvector_y,
        Bspxi, Bspeta, bspline_classification, extended_basis,
        local_inner,
    )
    return Ke, Fe, local_reduced


def boundaryElementWEB(model, p, q, knotvector_x, knotvector_y, elemx, elemy, x1, x2, y1, y2,
                       Bspxi, Bspeta, bspline_classification, extended_basis, basis_to_inner):
    """Process boundary element with WEB-splines (with subdivision)."""
    local_inner, local_reduced = _get_local_inner_dofs(elemx, elemy, p, q, bspline_classification, basis_to_inner)
    Ke, Fe = SubdivideWEB(
        model, x1, x2, y1, y2, p, q, knotvector_x, knotvector_y,
        Bspxi, Bspeta, bspline_classification, extended_basis,
        local_inner,
        level=0, MAXLEVEL=MAX_SUBDIVISION,
    )
    return Ke, Fe, local_reduced


def classifyAllElementsWEB(model, p, q, knotvector_x, knotvector_y, xDivision, yDivision):
    """
    Classify all elements (same as original, for determining subdivision needs).
    """
    all_corners = []
    elem_coords = []
    
    for elemx in range(p, p + xDivision + 1):
        for elemy in range(q, q + yDivision + 1):
            x1, x2 = knotvector_x[elemx], knotvector_x[elemx + 1]
            y1, y2 = knotvector_y[elemy], knotvector_y[elemy + 1]
            all_corners.extend([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            elem_coords.append((elemx, elemy, x1, x2, y1, y2))
    
    # Evaluate distances at corners using vectorized mesh function (ignore derivatives)
    all_corners_array = np.array(all_corners)
    x_coords = all_corners_array[:, 0]
    y_coords = all_corners_array[:, 1]
    distances_flat, _, _ = mesh.distance_with_derivative_vect_trasformed(
        x_coords, y_coords, model, transform=mesh.TRANSFORM
    )
    all_distances = distances_flat.view(-1, 4)
    
    min_d = all_distances.min(dim=1).values.detach().numpy()
    max_d = all_distances.max(dim=1).values.detach().numpy()
    
    inner_mask = min_d >= 0
    outer_mask = max_d < 0
    
    element_types = {"inner": [], "outer": [], "boundary": []}
    
    for idx, (elemx, elemy, x1, x2, y1, y2) in enumerate(elem_coords):
        if inner_mask[idx]:
            element_types["inner"].append((elemx, elemy, x1, x2, y1, y2))
        elif outer_mask[idx]:
            element_types["outer"].append((elemx, elemy, x1, x2, y1, y2))
        else:
            element_types["boundary"].append((elemx, elemy, x1, x2, y1, y2))
    
    return element_types


# =============================================================================
# MAIN ASSEMBLY FUNCTION
# =============================================================================

def processAllElementsWEB(
    model,
    p,
    q,
    knotvector_x,
    knotvector_y,
    xDivision,
    yDivision,
    extension_strict: bool = True,
    web_use_weight_normalization: bool = False,
    web_ref_weight_eps: float = 1e-10,
):
    """
    Process all elements using WEB-splines.
    
    This is the main entry point for WEB-spline assembly.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model representing the geometry (SDF)
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array-like
        Knot vectors
    xDivision, yDivision : int
        Number of divisions in each direction
    
    Returns:
    --------
    K : ndarray
        Global stiffness matrix (n_inner x n_inner)
    F : ndarray
        Global load vector (n_inner,)
    etype : dict
        Element type counts
    bspline_classification : dict
        B-spline classification info (needed for solution reconstruction)
    extended_basis : dict
        Extended basis structure (needed for solution reconstruction)
    """
    print("=" * 60)
    print("WEB-SPLINES (Weighted Extended B-splines) Assembly")
    print("=" * 60)
    
    # Step 1: Classify elements (inner/boundary/outer - Same as original)
    print("\n[1/4] Classifying elements...")
    element_types = classifyAllElementsWEB(model, p, q, knotvector_x, knotvector_y, xDivision, yDivision)

    etype = {
        "inner": len(element_types["inner"]),
        "outer": len(element_types["outer"]),
        "boundary": len(element_types["boundary"]),
    }

    print(f"      Inner elements:    {etype['inner']}")
    print(f"      Boundary elements: {etype['boundary']}")
    print(f"      Outer elements:    {etype['outer']}")

    # Step 2: Classify B-splines
    
    print("\n[2/4] Classifying B-splines (Höllig)...")
    bspline_classification = classifyBsplinesHollig(
        element_types, p, q, knotvector_x, knotvector_y, xDivision, yDivision
    )
    
    print(f"      Inner B-splines: {bspline_classification['n_inner']}")
    print(f"      Outer B-splines: {bspline_classification['n_outer']}")
    print(f"      Outside domain:  {len(bspline_classification['completely_outside'])}")

    # Optional: precompute w(x_i) reference values for Definition 2 normalization.
    bspline_classification['web_use_weight_normalization'] = bool(web_use_weight_normalization)
    if web_use_weight_normalization:
        print("\n[2.5/4] Precomputing reference weights w(x_i) for WEB normalization...")
        ref_weights, ref_points = _compute_inner_reference_weights_from_inner_elements(
            model,
            bspline_classification,
            element_types,
            p,
            q,
            eps=web_ref_weight_eps,
        )
        bspline_classification['inner_reference_weights'] = ref_weights
        bspline_classification['inner_reference_points'] = ref_points
        print(f"      Computed w(x_i) for {len(ref_weights)} inner B-splines")
    
    # Step 3: Compute extension coefficients using exact Höllig method (strict-only)
    print("\n[3/4] Computing extension coefficients (Höllig exact, strict)...")
    if not extension_strict:
        raise ValueError(
            "Non-strict WEB extension has been removed. Use extension_strict=True (strict Eq.(9) only)."
        )
    extension_coeffs = computeExtensionCoefficientsHollig(
        bspline_classification,
        p,
        q,
        debug=False,
        strict=True,
        element_types=element_types,
        xDivision=xDivision,
        yDivision=yDivision,
    )
    n_extended = sum(1 for v in extension_coeffs.values() if len(v) > 0)
    print(f"      Extended {n_extended} outer B-splines")
    
    # Step 4: Build extended basis
    print("\n[4/4] Building extended basis...")
    extended_basis = buildExtendedBasis(bspline_classification, extension_coeffs)

    # Build inverse map for correct local DOF selection:
    #   (basis index k) -> set of inner indices i such that k appears in B_i^e
    basis_to_inner = {}
    for inner_idx, basis_coeffs in extended_basis.items():
        for basis_idx in basis_coeffs.keys():
            basis_to_inner.setdefault(basis_idx, set()).add(inner_idx)
    
    print("\n      Assembling system...")
    
    # Get B-spline objects
    Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
    
    # Initialize reduced system (use sparse matrix for efficiency)
    n_inner = bspline_classification['n_inner']
    K = sparse.lil_matrix((n_inner, n_inner), dtype=np.float64)
    F = np.zeros(n_inner)
    
    total_elements = etype["inner"] + etype["boundary"]
    
    with tqdm(total=total_elements, desc="Processing elements", unit="elem") as pbar:
        # Process inner elements
        for elemx, elemy, x1, x2, y1, y2 in element_types["inner"]:
            Ke, Fe, ridxs = elementWEB(
                model, p, q, knotvector_x, knotvector_y,
                elemx, elemy, x1, x2, y1, y2,
                Bspxi, Bspeta, bspline_classification, extended_basis, basis_to_inner,
            )
            K, F = _assembly_web(K, F, Ke, Fe, ridxs)
            pbar.update(1)
        
        pbar.set_description("Processing boundary elements")
        
        # Process boundary elements
        for elemx, elemy, x1, x2, y1, y2 in element_types["boundary"]:
            Ke, Fe, ridxs = boundaryElementWEB(
                model, p, q, knotvector_x, knotvector_y,
                elemx, elemy, x1, x2, y1, y2,
                Bspxi, Bspeta, bspline_classification, extended_basis, basis_to_inner,
            )
            K, F = _assembly_web(K, F, Ke, Fe, ridxs)
            pbar.update(1)
    
    print(f"\n      System size: {n_inner} x {n_inner}")
    
    return K, F, etype, bspline_classification, extended_basis


# =============================================================================
# MATRIX TRANSFORMATION (Eq. 8.9)
# =============================================================================

def transformStandardSystemToWEB(
    K_full,
    F_full,
    model,
    p,
    q,
    knotvector_x,
    knotvector_y,
    xDivision,
    yDivision,
    extension_strict: bool = True,
    web_use_weight_normalization: bool = True,
    web_ref_weight_eps: float = 1e-10,
):
    """
    Apply the Höllig coupling-matrix transform (Eq. 8.9) to a standard
    weighted B-spline system assembled on the full basis.

    This keeps the numerical integration and assembly identical to the
    standard weighted B-spline code, then builds the WEB reduced system via
    	ilde{E} so that \hat{G} = \tilde{E} G \tilde{E}^T and \hat{F} = \tilde{E} F.
    """
    n_basis_x = len(knotvector_x) - p - 1
    n_basis_y = len(knotvector_y) - q - 1
    n_total_expected = n_basis_x * n_basis_y

    # Convert sparse matrix to dense if necessary
    if sparse.issparse(K_full):
        K_full = K_full.toarray()
    K_full = np.asarray(K_full, dtype=NP_DTYPE)
    F_full = np.asarray(F_full, dtype=NP_DTYPE)

    if K_full.shape != (n_total_expected, n_total_expected):
        raise ValueError(
            f"K_full has shape {K_full.shape}, expected {(n_total_expected, n_total_expected)} based on knot vectors and degrees."
        )
    if F_full.shape[0] != n_total_expected:
        raise ValueError(
            f"F_full has length {F_full.shape[0]}, expected {n_total_expected} based on knot vectors and degrees."
        )

    # Step 1: element and B-spline classification (same as WEB path)
    element_types = classifyAllElementsWEB(model, p, q, knotvector_x, knotvector_y, xDivision, yDivision)
    etype = {
        "inner": len(element_types["inner"]),
        "outer": len(element_types["outer"]),
        "boundary": len(element_types["boundary"]),
    }

    bspline_classification = classifyBsplinesHollig(
        element_types, p, q, knotvector_x, knotvector_y, xDivision, yDivision
    )

    # Optional WEB normalization term 1 / w(x_i)
    bspline_classification['web_use_weight_normalization'] = bool(web_use_weight_normalization)
    ref_weights = None
    if web_use_weight_normalization:
        ref_weights, ref_points = _compute_inner_reference_weights_from_inner_elements(
            model,
            bspline_classification,
            element_types,
            p,
            q,
            eps=web_ref_weight_eps,
        )
        bspline_classification['inner_reference_weights'] = ref_weights
        bspline_classification['inner_reference_points'] = ref_points

    # Step 2: compute extension coefficients and extended basis
    extension_coeffs = computeExtensionCoefficientsHollig(
        bspline_classification,
        p,
        q,
        debug=False,
        strict=extension_strict,
        element_types=element_types,
        xDivision=xDivision,
        yDivision=yDivision,
    )
    extended_basis = buildExtendedBasis(bspline_classification, extension_coeffs)

    # Step 3: build coupling matrix \tilde{E}
    n_inner = bspline_classification['n_inner']
    inner_to_reduced_idx = bspline_classification['inner_to_reduced_idx']
    E_tilde = np.zeros((n_inner, n_total_expected), dtype=NP_DTYPE)

    for inner_idx in bspline_classification['inner']:
        row = inner_to_reduced_idx[inner_idx]
        col = inner_idx[0] * n_basis_y + inner_idx[1]
        diag_scale = 1.0
        if ref_weights is not None:
            diag_scale = 1.0 / ref_weights[inner_idx]
        E_tilde[row, col] = diag_scale

    for outer_idx, coeffs in extension_coeffs.items():
        col = outer_idx[0] * n_basis_y + outer_idx[1]
        for inner_idx, coeff in coeffs.items():
            row = inner_to_reduced_idx.get(inner_idx)
            if row is not None:
                # If WEB normalization is enabled, scale the entire row i
                # (i.e. all coefficients contributing to φ_i) by 1 / w(x_i).
                if ref_weights is not None:
                    E_tilde[row, col] = (1.0 / ref_weights[inner_idx]) * coeff
                else:
                    E_tilde[row, col] = coeff

    # Step 4: transform system
    K_reduced = E_tilde @ K_full @ E_tilde.T
    F_reduced = E_tilde @ F_full

    return K_reduced, F_reduced, etype, bspline_classification, extended_basis, E_tilde


def transformStandardSystemToWEBSelectiveDiagonalExtraction(
    K_full,
    F_full,
    model,
    p,
    q,
    knotvector_x,
    knotvector_y,
    xDivision,
    yDivision,
    diag_threshold: float = 1e-10,
    diag_nonzero_eps: float = 0.0,
    extension_strict: bool = True,
    print_max: int = 25,
):
    """Selective coupling-matrix transform based on the stiffness diagonal.

    Compared to :func:`transformStandardSystemToWEB`, this only *extracts* (extends)
    those outer B-splines whose main diagonal entry |K[j,j]| is small-but-nonzero.

    - Outer B-splines with |K[j,j]| >= diag_threshold are kept as independent DOFs.
    - Outer B-splines with 0 < |K[j,j]| < diag_threshold are extracted via Höllig
      extension coefficients (same mechanism as WEB).
    - If 0 < |K[j,j]| < diag_threshold for an *inner* B-spline, it is kept (not
      extended) and reported to the user.

    Returns:
        (K_reduced, F_reduced, etype, meta, E_tilde)

    where meta contains reduced-basis bookkeeping.
    """
    n_basis_x = len(knotvector_x) - p - 1
    n_basis_y = len(knotvector_y) - q - 1
    n_total_expected = n_basis_x * n_basis_y

    # Convert sparse matrix to dense if necessary
    if sparse.issparse(K_full):
        K_full = K_full.toarray()
    K_full = np.asarray(K_full, dtype=NP_DTYPE)
    F_full = np.asarray(F_full, dtype=NP_DTYPE)

    if K_full.shape != (n_total_expected, n_total_expected):
        raise ValueError(
            f"K_full has shape {K_full.shape}, expected {(n_total_expected, n_total_expected)} based on knot vectors and degrees."
        )
    if F_full.shape[0] != n_total_expected:
        raise ValueError(
            f"F_full has length {F_full.shape[0]}, expected {n_total_expected} based on knot vectors and degrees."
        )

    if diag_threshold <= 0:
        raise ValueError("diag_threshold must be > 0")
    if diag_nonzero_eps < 0:
        raise ValueError("diag_nonzero_eps must be >= 0")

    # Step 1: element and B-spline classification
    element_types = classifyAllElementsWEB(model, p, q, knotvector_x, knotvector_y, xDivision, yDivision)
    etype = {
        "inner": len(element_types["inner"]),
        "outer": len(element_types["outer"]),
        "boundary": len(element_types["boundary"]),
    }

    bspline_classification = classifyBsplinesHollig(
        element_types, p, q, knotvector_x, knotvector_y, xDivision, yDivision
    )
    inner_set = set(bspline_classification["inner"])
    outer_set = set(bspline_classification["outer"])
    print(f"Total B-splines: {n_total_expected}, Inner: {len(inner_set)}, Outer: {len(outer_set)}")

    # Step 2: decide which outer B-splines to extract based on K diagonal
    diag = np.diag(K_full)
    abs_diag = np.abs(diag)
    small_nonzero_mask = (abs_diag < diag_threshold) & (abs_diag > diag_nonzero_eps)

    extracted_outer = set()
    kept_outer = set()
    small_inner = []
    small_other = []

    for col in np.where(small_nonzero_mask)[0]:
        idx = (int(col // n_basis_y), int(col % n_basis_y))
        if idx in outer_set:
            extracted_outer.add(idx)
        elif idx in inner_set:
            small_inner.append((idx, float(diag[col])))
        else:
            small_other.append((idx, float(diag[col])))

    kept_outer = outer_set - extracted_outer

    if small_inner:
        print("*** NOTE: small-but-nonzero diagonal entries detected for INNER B-splines (kept as-is):")
        for (idx, val) in small_inner[: max(0, int(print_max))]:
            print(f"    inner {idx}: K_jj = {val:.3e}")
        if len(small_inner) > print_max:
            print(f"    ... and {len(small_inner) - print_max} more")

    if small_other:
        print("*** NOTE: small-but-nonzero diagonal entries for NON-inner/non-outer indices (ignored):")
        for (idx, val) in small_other[: max(0, int(print_max))]:
            print(f"    idx {idx}: K_jj = {val:.3e}")
        if len(small_other) > print_max:
            print(f"    ... and {len(small_other) - print_max} more")

    print(
        f"Selective extraction: extracting {len(extracted_outer)}/{len(outer_set)} outer B-splines "
        f"with |K_jj| < {diag_threshold:g} (and > {diag_nonzero_eps:g}); keeping {len(kept_outer)} outer DOFs."
    )

    # Step 3: compute extension coefficients only for extracted outer set
    if len(extracted_outer) > 0:
        bspline_classification_sel = dict(bspline_classification)
        bspline_classification_sel["outer"] = list(extracted_outer)
        extension_coeffs = computeExtensionCoefficientsHollig(
            bspline_classification_sel,
            p,
            q,
            debug=False,
            strict=extension_strict,
            element_types=element_types,
            xDivision=xDivision,
            yDivision=yDivision,
        )
    else:
        extension_coeffs = {}

    # Step 4: build coupling matrix for a mixed reduced basis
    reduced_basis = list(bspline_classification["inner"]) + sorted(kept_outer)
    reduced_to_row = {idx: r for r, idx in enumerate(reduced_basis)}
    n_reduced = len(reduced_basis)

    E_tilde = np.zeros((n_reduced, n_total_expected), dtype=NP_DTYPE)

    # identity rows for kept dofs
    for idx in reduced_basis:
        col = idx[0] * n_basis_y + idx[1]
        row = reduced_to_row[idx]
        E_tilde[row, col] = 1.0

    # overwrite extracted outer columns by their inner combinations
    for outer_idx, coeffs in extension_coeffs.items():
        col = outer_idx[0] * n_basis_y + outer_idx[1]
        # zero out the identity entry if it exists (it shouldn't for extracted outers)
        E_tilde[:, col] = 0.0
        for inner_idx, coeff in coeffs.items():
            row = reduced_to_row.get(inner_idx)
            if row is not None:
                E_tilde[row, col] = float(coeff)

    # Step 5: transform system
    K_reduced = E_tilde @ K_full @ E_tilde.T
    F_reduced = E_tilde @ F_full

    meta = {
        "reduced_basis": reduced_basis,
        "reduced_to_row": reduced_to_row,
        "inner_basis": list(bspline_classification["inner"]),
        "kept_outer_basis": sorted(kept_outer),
        "extracted_outer_basis": sorted(extracted_outer),
        "diag_threshold": float(diag_threshold),
        "diag_nonzero_eps": float(diag_nonzero_eps),
    }

    return K_reduced, F_reduced, etype, meta, E_tilde


# =============================================================================
# SOLVER
# =============================================================================

def solveWEB(K, F):
    """
    Solve the WEB-spline system using sparse or dense matrix format.
    
    Since we only have inner DOFs, the system should be well-conditioned.
    Handles both sparse matrices (from processAllElementsWEB) and dense arrays
    (from matrix transformation functions).
    """
    # Convert to csr format for efficient operations (or handle if already dense)
    if sparse.issparse(K):
        K_csr = K.tocsr()
    else:
        # K is a dense array (from transformation functions)
        K_dense = np.asarray(K, dtype=NP_DTYPE)
        K_csr = sparse.csr_matrix(K_dense)
    
    # Check for zero rows (shouldn't happen with WEB-splines, but be safe)
    non_zero_rows = np.array(K_csr.getnnz(axis=1) > 0)
    
    if not np.any(non_zero_rows):
        return np.zeros(len(F))
    
    if np.all(non_zero_rows):
        # Full system - use direct solve
        K_solve = K_csr.toarray()
        F_solve = F
        full_solve = True
    else:
        # Some zero rows - reduce system
        K_solve = K_csr[non_zero_rows, :][:, non_zero_rows].toarray()
        F_solve = F[non_zero_rows]
        full_solve = False
    
    # Report condition number
    try:
        pass
        #cond = np.linalg.cond(K_solve)
        #print(f"Condition number: {cond:.2e}")
    except:
        pass
    
    # Solve
    try:
        u_solve = linalg.solve(K_solve, F_solve, assume_a='sym')
    except:
        u_solve = np.linalg.solve(K_solve, F_solve)
    
    print(f"Max residual: {np.max(np.abs(K_solve @ u_solve - F_solve)):.2e}")
    
    if full_solve:
        return u_solve
    else:
        u = np.zeros(len(F))
        u[non_zero_rows] = u_solve
        return u


# =============================================================================
# SOLUTION RECONSTRUCTION
# =============================================================================

def reconstructSolution(x, y, u_reduced, model, p, q, knotvector_x, knotvector_y,
                        bspline_classification, extended_basis):
    """
    Reconstruct the solution at a point (x, y) from WEB-spline coefficients.
    
    The solution is:
        u(x,y) = d(x,y) * Σ_i u_i * B_i^e(x,y) + (1 - d(x,y)) * g(x,y)
    
    where B_i^e is the extended B-spline.
    
    Parameters:
    -----------
    x, y : float
        Evaluation point
    u_reduced : array
        Solution coefficients (inner DOFs only)
    model : neural network
        SDF model
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array
        Knot vectors
    bspline_classification : dict
        B-spline classification info
    extended_basis : dict
        Extended basis structure
    
    Returns:
    --------
    float : Solution value at (x, y)
    """
    # Get distance
    d = mesh.distanceFromContur(x, y, model)
    if hasattr(d, 'item'):
        d = d.item()
    elif hasattr(d, 'detach'):
        d = d.detach().numpy().item()
    else:
        d = float(d)
    
    if d < 0:
        return 0.0  # Outside domain
    
    # Evaluate B-splines
    Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
    bxi = Bspxi(x)
    beta = Bspeta(y)
    
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']
    inner_bsplines = bspline_classification['inner']
    
    # Compute (possibly normalized) extended B-spline sum
    # If WEB normalization is enabled, the effective basis is (d / w(x_i)) * B_i^e.
    bspline_sum = 0.0
    use_norm = bool(bspline_classification.get('web_use_weight_normalization', False))
    ref_weights = bspline_classification.get('inner_reference_weights') if use_norm else None
    
    for reduced_idx, inner_idx in enumerate(inner_bsplines):
        coeff = u_reduced[reduced_idx]
        basis_coeffs = extended_basis[inner_idx]
        
        # Evaluate extended B-spline
        Be_value = 0.0
        for (bi, bj), ecoeff in basis_coeffs.items():
            if bi < n_basis_x and bj < n_basis_y:
                Be_value += ecoeff * bxi[bi] * beta[bj]

        if use_norm:
            w_i = ref_weights.get(inner_idx) if ref_weights is not None else None
            if w_i is None:
                raise RuntimeError(
                    "WEB normalization enabled but missing w(x_i) for inner index "
                    f"{inner_idx}."
                )
            Be_value = Be_value / float(w_i)

        bspline_sum += coeff * Be_value
    
    # Get Dirichlet boundary value
    g = dirichletBoundary_vectorized(x, y)
    if hasattr(g, 'item'):
        g = g.item()
    elif isinstance(g, np.ndarray):
        g = float(g.flat[0]) if g.size > 0 else float(g)
    else:
        g = float(g)
    
    # Final solution
    u = d * bspline_sum + (1 - d) * g
    
    return u


def reconstructSolutionGradient(x, y, u_reduced, model, p, q, knotvector_x, knotvector_y,
                                 bspline_classification, extended_basis):
    """
    Reconstruct the solution gradient at a point (x, y) from WEB-spline coefficients.
    
    Returns:
    --------
    tuple : (du/dx, du/dy) at (x, y)
    """
    # Get distance and its derivatives
    d_tensor, dx_d, dy_d = mesh.distance_with_derivative(x, y, model)
    
    # Convert d_tensor to float
    if hasattr(d_tensor, 'item'):
        d = d_tensor.item()
    elif hasattr(d_tensor, 'detach'):
        d = d_tensor.detach().numpy().item()
    else:
        d = float(d_tensor)
    
    # Convert dx_d to float
    if hasattr(dx_d, 'item'):
        dx_d = dx_d.item()
    elif hasattr(dx_d, 'detach'):
        dx_d = dx_d.detach().numpy().item()
    else:
        dx_d = float(dx_d)
    
    # Convert dy_d to float
    if hasattr(dy_d, 'item'):
        dy_d = dy_d.item()
    elif hasattr(dy_d, 'detach'):
        dy_d = dy_d.detach().numpy().item()
    else:
        dy_d = float(dy_d)
    
    if d < 0:
        return 0.0, 0.0  # Outside domain
    
    # Evaluate B-splines and their derivatives
    Bspxi, Bspeta = _get_bspline_objects(knotvector_x, knotvector_y, p, q)
    bxi = Bspxi(x)
    beta = Bspeta(y)
    dbxi = Bspxi.diff(1)(x)
    dbeta = Bspeta.diff(1)(y)
    
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']
    inner_bsplines = bspline_classification['inner']
    
    # Compute (possibly normalized) extended B-spline sums
    bspline_sum = 0.0
    bspline_sum_dx = 0.0
    bspline_sum_dy = 0.0

    use_norm = bool(bspline_classification.get('web_use_weight_normalization', False))
    ref_weights = bspline_classification.get('inner_reference_weights') if use_norm else None
    
    for reduced_idx, inner_idx in enumerate(inner_bsplines):
        coeff = u_reduced[reduced_idx]
        basis_coeffs = extended_basis[inner_idx]
        
        Be_value = 0.0
        dBe_dx = 0.0
        dBe_dy = 0.0
        
        for (bi, bj), ecoeff in basis_coeffs.items():
            if bi < n_basis_x and bj < n_basis_y:
                Be_value += ecoeff * bxi[bi] * beta[bj]
                dBe_dx += ecoeff * dbxi[bi] * beta[bj]
                dBe_dy += ecoeff * bxi[bi] * dbeta[bj]

        if use_norm:
            w_i = ref_weights.get(inner_idx) if ref_weights is not None else None
            if w_i is None:
                raise RuntimeError(
                    "WEB normalization enabled but missing w(x_i) for inner index "
                    f"{inner_idx}."
                )
            inv_w = 1.0 / float(w_i)
            Be_value *= inv_w
            dBe_dx *= inv_w
            dBe_dy *= inv_w

        bspline_sum += coeff * Be_value
        bspline_sum_dx += coeff * dBe_dx
        bspline_sum_dy += coeff * dBe_dy
    
    # Get Dirichlet boundary values and derivatives
    g = dirichletBoundary_vectorized(x, y)
    dg_dx = dirichletBoundaryDerivativeX_vectorized(x, y)
    dg_dy = dirichletBoundaryDerivativeY_vectorized(x, y)
    
    # Convert to float
    if hasattr(g, 'item'):
        g = g.item()
    elif isinstance(g, np.ndarray):
        g = float(g.flat[0]) if g.size > 0 else float(g)
    else:
        g = float(g)
    
    if hasattr(dg_dx, 'item'):
        dg_dx = dg_dx.item()
    elif isinstance(dg_dx, np.ndarray):
        dg_dx = float(dg_dx.flat[0]) if dg_dx.size > 0 else float(dg_dx)
    else:
        dg_dx = float(dg_dx)
    
    if hasattr(dg_dy, 'item'):
        dg_dy = dg_dy.item()
    elif isinstance(dg_dy, np.ndarray):
        dg_dy = float(dg_dy.flat[0]) if dg_dy.size > 0 else float(dg_dy)
    else:
        dg_dy = float(dg_dy)
    
    # Gradient using product rule
    du_dx = dx_d * (bspline_sum - g) + d * bspline_sum_dx + (1 - d) * dg_dx
    du_dy = dy_d * (bspline_sum - g) + d * bspline_sum_dy + (1 - d) * dg_dy
    
    return du_dx, du_dy


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    import time
    from Geomertry import AnaliticalDistanceCircle
    
    print("\n" + "=" * 60)
    print("WEB-SPLINES DEMO")
    print("=" * 60)
    
    # Simple test with analytical circle
    model = AnaliticalDistanceCircle()
    
    DIVISIONS = 10
    ORDER = 1
    DELTA = 0.005
    
    default = mesh.getDefaultValues(div=DIVISIONS, order=ORDER, delta=DELTA)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knotvector_u, knotvector_w, weights, ctrlpts = mesh.generateRectangularMesh(*default)
    
    print(f"\nTest configuration:")
    print(f"  Divisions: {DIVISIONS}")
    print(f"  Order: {ORDER}")
    print(f"  DOFs (standard): {(xDivision + p + 1) * (yDivision + q + 1)}")
    
    start = time.time()
    K, F, etype, bsp_class, ext_basis = processAllElementsWEB(
        model, p, q, knotvector_u, knotvector_w, xDivision, yDivision
    )
    assembly_time = time.time() - start
    
    print(f"\nAssembly time: {assembly_time:.2f}s")
    print(f"Element types: {etype}")
    
    start = time.time()
    u = solveWEB(K, F)
    solve_time = time.time() - start
    
    print(f"Solve time: {solve_time:.4f}s")
    print(f"DOFs (WEB): {len(u)}")
    
    # Test solution reconstruction
    test_x, test_y = 0.3, 0.4
    u_val = reconstructSolution(test_x, test_y, u, model, p, q, 
                                knotvector_u, knotvector_w, bsp_class, ext_basis)
    u_exact = solution_function(test_x, test_y)
    
    print(f"\nSolution at ({test_x}, {test_y}):")
    print(f"  WEB-spline: {u_val:.6f}")
    print(f"  Exact:      {u_exact:.6f}")
    print(f"  Error:      {abs(u_val - u_exact):.2e}")
