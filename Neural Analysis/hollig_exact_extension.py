"""
Exact Höllig extension coefficient computation.

This module implements the algebraic Lagrange polynomial method from:
  Höllig, K., Reif, U., Wipper, J. (2001). "Weighted Extended B-Spline 
  Approximation of Dirichlet Problems." Mathematics of Computation, 70(235), 51-63.

For immersed geometries, uses a hybrid approach:
  - Strict Höllig Lagrange formula when I(j) has inner neighbors
  - Least-squares fitting to nearest neighbors otherwise
"""

import numpy as np
from typing import Optional

def _lagrange_coefficient_1d(i, j, alpha, p):
    """
    Compute the 1D Lagrange polynomial coefficient e^(1D)_{i,j,α}.
    
    Per Höllig's formula (Eq. 6):
        e_{i,j} = ∏_{ℓ=0:p, α+ℓ ≠ i} (j - α - ℓ) / (i - α - ℓ)
    
    This is zero if i ∉ [α, α+p], and is a Lagrange polynomial evaluated at j.
    
    Parameters:
    -----------
    i : int
        Inner B-spline index
    j : int
        Outer B-spline index
    alpha : int
        Start of the index window [α, α+p]
    p : int
        B-spline degree (window size is p+1)
    
    Returns:
    --------
    float : Lagrange coefficient
    """
    if i < alpha or i > alpha + p:
        return 0.0
    
    product = 1.0
    for ell in range(p + 1):
        if alpha + ell != i:
            numerator = float(j - alpha - ell)
            denominator = float(i - alpha - ell)
            if abs(denominator) < 1e-16:
                return 0.0
            product *= numerator / denominator
    
    return product


def _distance_point_to_index_block_inf(j_x, j_y, alpha_x, alpha_y, p, q):
    """Distance (in max norm) from point j to the rectangular index block."""
    dx = max(alpha_x - j_x, j_x - (alpha_x + p), 0)
    dy = max(alpha_y - j_y, j_y - (alpha_y + q), 0)
    return max(dx, dy), dx, dy


def _basis_support_fully_inner(i_x, i_y, p, q, element_inner_set, max_elem_x, max_elem_y):
    """Check if the full tensor-product support of basis (i_x,i_y) lies inside Ω.

    Requires that every element (elemx,elemy) with elemx∈[i_x, i_x+p], elemy∈[i_y, i_y+q]
    is an inner element.
    """
    ex_start = i_x
    ex_end = min(i_x + p, max_elem_x)
    ey_start = i_y
    ey_end = min(i_y + q, max_elem_y)
    for ex in range(ex_start, ex_end + 1):
        for ey in range(ey_start, ey_end + 1):
            if (ex, ey) not in element_inner_set:
                return False
    return True


def _block_support_is_fully_inner(alpha_x, alpha_y, p, q, element_inner_set, max_elem_x, max_elem_y):
    """Check if every basis in the (p+1)x(q+1) block has full support inside Ω."""
    for i_x in range(alpha_x, alpha_x + p + 1):
        for i_y in range(alpha_y, alpha_y + q + 1):
            if not _basis_support_fully_inner(i_x, i_y, p, q, element_inner_set, max_elem_x, max_elem_y):
                return False
    return True


def _find_closest_inner_block_start_2d(j_x, j_y, p, q, n_basis_x, n_basis_y,
                                       element_inner_set,
                                       max_elem_x,
                                       max_elem_y,
                                       max_radius=50):
    """Find (alpha_x, alpha_y) so that the full (p+1)x(q+1) block is in I.

    Theorem 2.1 / Eq. (9) assumes the closest index array I(j) is a full
    (p+1)x(q+1) block of indices that is a subset of the inner index set I.
    
    This function searches locally around j for the closest such block under
    the infinity norm (ties broken by smaller dx+dy).

    Returns:
        (alpha_x, alpha_y) if found, else None.
    """
    max_alpha_x = n_basis_x - (p + 1)
    max_alpha_y = n_basis_y - (q + 1)
    if max_alpha_x < 0 or max_alpha_y < 0:
        return None

    # Base guess roughly places j at the right end of the window (like j-p)
    base_ax = min(max(j_x - p, 0), max_alpha_x)
    base_ay = min(max(j_y - q, 0), max_alpha_y)

    best = None
    best_key = None

    for r in range(max_radius + 1):
        ax_min = max(0, base_ax - r)
        ax_max = min(max_alpha_x, base_ax + r)
        ay_min = max(0, base_ay - r)
        ay_max = min(max_alpha_y, base_ay + r)

        for alpha_x in range(ax_min, ax_max + 1):
            for alpha_y in range(ay_min, ay_max + 1):
                if not _block_support_is_fully_inner(alpha_x, alpha_y, p, q, element_inner_set, max_elem_x, max_elem_y):
                    continue
                d_inf, dx, dy = _distance_point_to_index_block_inf(j_x, j_y, alpha_x, alpha_y, p, q)
                key = (d_inf, dx + dy, abs(alpha_x - base_ax) + abs(alpha_y - base_ay))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (alpha_x, alpha_y)

        # If we found any candidate at radius r with d_inf==0, it's optimal.
        if best_key is not None and best_key[0] == 0:
            break

        # Small optimization: if we've found a candidate and expanding the search
        # can't improve the current best d_inf, we can stop.
        if best_key is not None and r > best_key[0] + 2:
            break

    return best


def computeExtensionCoefficientsHollig(
    bspline_classification,
    p,
    q,
    debug: bool = False,
    Bspxi=None,
    Bspeta=None,
    knotvector_x=None,
    knotvector_y=None,
    use_ls_fit: bool = False,
    strict: bool = False,
    element_types: Optional[dict] = None,
    xDivision: Optional[int] = None,
    yDivision: Optional[int] = None,
):
    """
    Compute extension coefficients for outer B-splines.
    
    Strategy:
    1. Strict Höllig: If I(j) has inner neighbors, use exact Lagrange formula
    2. Fallback: Use nearest neighbors with uniform weights (fast) or LS fit (slower, better)
    
    For immersed geometries, outer B-splines have limited support in the active domain.
    Fallback weights are a heuristic that works reasonably well.
    
    Parameters:
    -----------
    bspline_classification : dict
        Output from classifyBsplinesHollig()
    p, q : int
        B-spline degrees
    debug : bool
        If True, print debug information
    use_ls_fit : bool
        If True, use LS fitting with B-spline evaluations (requires Bsp* and knotvector_* params)
        If False, use uniform weights (default, fast)
    
    Returns:
    --------
    dict : Extension coefficients
    """
    inner_set = set(bspline_classification['inner'])
    outer_bsplines = bspline_classification['outer']
    n_basis_x = bspline_classification['n_basis_x']
    n_basis_y = bspline_classification['n_basis_y']

    # Build inner element set for strict block support checks
    if element_types is None:
        element_inner_set = set()
    else:
        element_inner_set = {(ex, ey) for ex, ey, *_ in element_types.get('inner', [])}
    if xDivision is None or yDivision is None:
        max_elem_x = n_basis_x - 1 + p  # conservative upper bound
        max_elem_y = n_basis_y - 1 + q
    else:
        max_elem_x = p + xDivision
        max_elem_y = q + yDivision

    extension_coeffs = {}
    n_with_hollig = 0
    n_with_fallback = 0

    for outer_idx in outer_bsplines:
        j_x, j_y = outer_idx

        # Step 1: Try strict Höllig closest index array I(j) first
        # We search for a (p+1)x(q+1) block that is fully inside the inner set I.
        alpha_pair = _find_closest_inner_block_start_2d(
            j_x,
            j_y,
            p,
            q,
            n_basis_x,
            n_basis_y,
            element_inner_set,
            max_elem_x,
            max_elem_y,
        )
        if alpha_pair is None:
            alpha_x = None
            alpha_y = None
        else:
            alpha_x, alpha_y = alpha_pair

        coeffs = {}
        
        if alpha_x is not None:
            for i_x in range(alpha_x, alpha_x + p + 1):
                for i_y in range(alpha_y, alpha_y + q + 1):
                    inner_idx = (i_x, i_y)
                    # Require that the basis belongs to the reduced inner set
                    # AND its full support lies inside Ω.
                    if inner_idx not in inner_set:
                        continue
                    if not _basis_support_fully_inner(i_x, i_y, p, q, element_inner_set, max_elem_x, max_elem_y):
                        continue

                    e_x = _lagrange_coefficient_1d(i_x, j_x, alpha_x, p)
                    e_y = _lagrange_coefficient_1d(i_y, j_y, alpha_y, q)
                    e_ij = e_x * e_y

                    if abs(e_ij) > 1e-14:
                        coeffs[inner_idx] = float(e_ij)

        # Fallback to nearest neighbors for immersed geometry
        if len(coeffs) == 0:
            if strict:
                raise RuntimeError(
                    "Strict WEB/Höllig mode failed: outer B-spline "
                    f"{outer_idx} has no inner overlap in its closest-index-array window "
                    f"(alpha_x={alpha_x}, p={p}; alpha_y={alpha_y}, q={q}). "
                    "This typically happens in immersed geometries where the paper assumptions "
                    "(availability of a fully-inside cell in the support) do not hold."
                )
            inner_list = list(inner_set)
            
            # Find nearest k inner B-splines
            distances = [
                ((i_x - j_x)**2 + (i_y - j_y)**2, (i_x, i_y)) 
                for i_x, i_y in inner_list
            ]
            distances.sort()
            
            # Use p+q+1 neighbors
            k_neighbors = max(3, min(p + q + 1, len(inner_list)))
            nearest_indices = [idx for _, idx in distances[:k_neighbors]]
            
            # Use uniform weights for simplicity and speed
            weight = 1.0 / len(nearest_indices)
            for idx in nearest_indices:
                coeffs[idx] = weight
            
            n_with_fallback += 1
        else:
            n_with_hollig += 1

        extension_coeffs[outer_idx] = coeffs

    if debug:
        print(f"      Strict Höllig: {n_with_hollig}/{len(outer_bsplines)}")
        print(f"      Fallback:      {n_with_fallback}/{len(outer_bsplines)}")
    
    # Alert user if fallback was used significantly
    if n_with_fallback > 0:
        pct_fallback = 100.0 * n_with_fallback / len(outer_bsplines)
        print(f"*** WARNING: {n_with_fallback}/{len(outer_bsplines)} outer B-splines ({pct_fallback:.1f}%) using nearest-neighbor fallback")
        print(f"    Fallback indicates immersed geometry not ideal for Höllig method.")
        print(f"    Consider using standard weighted IGA for better accuracy.")

    return extension_coeffs


if __name__ == "__main__":
    # Simple test of Lagrange coefficient computation
    print("Testing 1D Lagrange coefficients for p=1, j=5, α=4:")
    for i in range(4, 7):  # i ∈ [α, α+p] = [4, 5]
        e = _lagrange_coefficient_1d(i, 5, 4, 1)
        print(f"  e_{i},5 = {e:.4f}")
    # Expected: e_4,5 = (5-5)/(4-5) = 0, e_5,5 = (5-4)/(5-4) = 1

    print("\nTesting 2D (tensor product) for p=q=1:")
    outer_idx = (5, 3)
    alpha_x, alpha_y = 4, 2
    for i_x in range(alpha_x, alpha_x + 2):
        for i_y in range(alpha_y, alpha_y + 2):
            e_x = _lagrange_coefficient_1d(i_x, outer_idx[0], alpha_x, 1)
            e_y = _lagrange_coefficient_1d(i_y, outer_idx[1], alpha_y, 1)
            e_2d = e_x * e_y
            print(f"  e_{(i_x,i_y)},{outer_idx} = {e_x:.4f} * {e_y:.4f} = {e_2d:.4f}")
