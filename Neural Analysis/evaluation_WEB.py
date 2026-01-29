"""
Evaluation utilities for WEB-splines.

Provides functions to evaluate accuracy and visualize results
from WEB-spline simulations.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from bspline import Bspline

import FEM
import FEM_WEB
import mesh

TORCH_DTYPE = torch.float64


def visualizeReconstructionAtPointWEB(
    model,
    x: float,
    y: float,
    u_reduced,
    p: int,
    q: int,
    knotvector_x,
    knotvector_y,
    bspline_classification,
    extended_basis,
    *,
    top_k: int = 25,
    contribution_tol: float = 0.0,
    show: bool = True,
):
    """Visualize how the WEB solution is reconstructed at a single point.

    This is meant for debugging/understanding the reconstruction:
      u(x,y) = d(x,y) * \sum_i u_i * B_i^e(x,y) + (1 - d(x,y)) * g(x,y)

    where the WEB basis is (optionally) normalized as (d / w(x_i)) * B_i^e.

    What this function shows:
      - which tensor-product B-spline indices (i,j) are active at (x,y)
      - which of those are inner/outer according to the WEB classification
      - per-inner-DOF effective extended value B_i^e(x,y) and its contribution
      - a bar plot of the largest |contribution| terms

    Notes on efficiency:
      We avoid iterating the full extended-basis dictionary for every i by
      only mixing in *active* outer indices at (x,y).

    Returns:
      details: dict with numerical breakdown (useful for logging/tests).
    """
    # Distance + boundary value
    d = mesh.distanceFromContur(x, y, model)
    if hasattr(d, 'item'):
        d = d.item()
    elif hasattr(d, 'detach'):
        d = d.detach().cpu().numpy().item()
    else:
        d = float(d)

    g = FEM.dirichletBoundary_vectorized(x, y)
    if hasattr(g, 'item'):
        g = g.item()
    elif isinstance(g, np.ndarray):
        g = float(g.flat[0]) if g.size > 0 else float(g)
    else:
        g = float(g)

    details = {
        'x': float(x),
        'y': float(y),
        'd': float(d),
        'g': float(g),
        'outside': bool(d < 0),
    }

    if d < 0:
        # Outside Ω: by convention reconstruction returns 0.0 in FEM_WEB
        details.update({'bspline_sum': 0.0, 'u_total': 0.0, 'active_pairs': []})
        return details

    # Evaluate tensor-product B-spline values at point
    Bspxi, Bspeta = FEM._get_bspline_objects(knotvector_x, knotvector_y, p, q)
    bxi = np.asarray(Bspxi(x), dtype=np.float64)
    beta = np.asarray(Bspeta(y), dtype=np.float64)

    nz_x = np.where(np.abs(bxi) > 0)[0]
    nz_y = np.where(np.abs(beta) > 0)[0]

    active_pairs = [(int(i), int(j)) for i in nz_x for j in nz_y if (bxi[i] != 0.0 and beta[j] != 0.0)]
    details['active_pairs'] = active_pairs

    inner_set = set(map(tuple, bspline_classification.get('inner', [])))
    outer_set = set(map(tuple, bspline_classification.get('outer', [])))

    active_inner_pairs = [ij for ij in active_pairs if ij in inner_set]
    active_outer_pairs = [ij for ij in active_pairs if ij in outer_set]
    details['active_inner_pairs'] = active_inner_pairs
    details['active_outer_pairs'] = active_outer_pairs

    use_norm = bool(bspline_classification.get('web_use_weight_normalization', False))
    ref_weights = bspline_classification.get('inner_reference_weights') if use_norm else None

    # Helper to evaluate tensor-product B-spline value b_{i,j}(x,y)
    def _b_val(pair):
        ii, jj = pair
        if ii < 0 or jj < 0 or ii >= len(bxi) or jj >= len(beta):
            return 0.0
        return float(bxi[ii] * beta[jj])

    # Contributions per inner DOF
    inner_bsplines = bspline_classification.get('inner', [])
    contributions = []
    bspline_sum = 0.0

    # Precompute active outer basis values once
    active_outer_values = {pair: _b_val(pair) for pair in active_outer_pairs}

    for reduced_idx, inner_idx in enumerate(inner_bsplines):
        coeff = float(u_reduced[reduced_idx])

        # B_i^e(x,y) = b_i(x,y) + sum_{j in active_outer} e_{j,i} * b_j(x,y)
        Be = _b_val(inner_idx)
        basis_coeffs = extended_basis.get(inner_idx)
        if basis_coeffs is None:
            raise KeyError(f"Missing extended basis entry for inner index {inner_idx}.")

        if active_outer_values:
            for outer_idx, b_outer in active_outer_values.items():
                if b_outer == 0.0:
                    continue
                e_ji = basis_coeffs.get(outer_idx)
                if e_ji is None or e_ji == 0.0:
                    continue
                Be += float(e_ji) * float(b_outer)

        norm_factor = 1.0
        if use_norm:
            if ref_weights is None:
                raise RuntimeError("WEB normalization enabled but bspline_classification has no 'inner_reference_weights'.")
            w_i = ref_weights.get(inner_idx)
            if w_i is None:
                raise RuntimeError(f"WEB normalization enabled but missing w(x_i) for inner index {inner_idx}.")
            norm_factor = float(w_i)
            Be = Be / norm_factor

        phi_i = float(d) * float(Be)
        contrib = coeff * phi_i
        bspline_sum += coeff * float(Be)

        if abs(contrib) > contribution_tol or abs(Be) > 0.0:
            contributions.append({
                'reduced_idx': int(reduced_idx),
                'inner_idx': tuple(inner_idx),
                'u_i': coeff,
                'Be': float(Be),
                'phi_i': float(phi_i),
                'contribution': float(contrib),
                'w_i': float(norm_factor) if use_norm else None,
            })

    u_total = float(d) * float(bspline_sum) + (1.0 - float(d)) * float(g)
    details['bspline_sum'] = float(bspline_sum)
    details['u_total'] = float(u_total)
    details['use_norm'] = bool(use_norm)
    details['n_contrib_nontrivial'] = int(len(contributions))

    # Sort and select top-k
    contributions_sorted = sorted(contributions, key=lambda r: abs(r['contribution']), reverse=True)
    top = contributions_sorted[: max(int(top_k), 0)]
    details['top_contributions'] = top

    # ---- Plot ----
    if show:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.4])

        # Left: Full domain solution field with point marked
        ax_domain = fig.add_subplot(gs[0, 0])
        ax_domain.set_title("Domain solution field u(x,y)")
        
        # Sample solution over a grid
        marg = 0.05
        x_grid = np.linspace(FEM.DOMAIN["x1"] - marg, FEM.DOMAIN["x2"] + marg, 60)
        y_grid = np.linspace(FEM.DOMAIN["y1"] - marg, FEM.DOMAIN["y2"] + marg, 60)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_grid = np.zeros_like(X_grid)
        
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                xx, yy = x_grid[i], y_grid[j]
                d_ij = mesh.distanceFromContur(xx, yy, model)
                if hasattr(d_ij, 'item'):
                    d_ij = d_ij.item()
                elif hasattr(d_ij, 'detach'):
                    d_ij = d_ij.detach().cpu().numpy().item()
                else:
                    d_ij = float(d_ij)
                
                if d_ij >= 0:
                    Z_grid[j, i] = FEM_WEB.reconstructSolution(
                        xx, yy, u_reduced, model, p, q,
                        knotvector_x, knotvector_y,
                        bspline_classification, extended_basis
                    )
                else:
                    Z_grid[j, i] = np.nan
        
        im_domain = ax_domain.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap='viridis')
        plt.colorbar(im_domain, ax=ax_domain, label='u(x,y)')
        ax_domain.scatter([x], [y], s=300, marker='*', c='red', edgecolors='white', linewidths=1.5, label=f'Point ({x:.3g}, {y:.3g})')
        ax_domain.set_xlabel("x")
        ax_domain.set_ylabel("y")
        ax_domain.legend(loc='upper right', fontsize=9)
        ax_domain.set_aspect('equal', adjustable='box')
        ax_domain.grid(True, alpha=0.2)

        # Middle: Active tensor-product indices
        ax0 = fig.add_subplot(gs[0, 1])
        ax0.set_title("Active basis indices at (x,y)")
        if len(active_pairs) == 0:
            ax0.text(0.5, 0.5, "No active basis?!", ha='center', va='center')
        else:
            # Plot all basis indices faintly
            all_inner = np.asarray(inner_set, dtype=int)
            all_outer = np.asarray(outer_set, dtype=int)
            if len(all_inner) > 0:
                ax0.scatter(all_inner[:, 0], all_inner[:, 1], s=20, c='tab:blue', alpha=0.2)
            if len(all_outer) > 0:
                ax0.scatter(all_outer[:, 0], all_outer[:, 1], s=20, c='tab:orange', alpha=0.2)
            
            # Highlight active pairs
            xs = [ij[0] for ij in active_pairs]
            ys = [ij[1] for ij in active_pairs]
            colors = [
                'tab:blue' if ij in inner_set else ('tab:orange' if ij in outer_set else 'tab:gray')
                for ij in active_pairs
            ]
            ax0.scatter(xs, ys, s=150, c=colors, edgecolors='k', linewidths=0.8, zorder=10)
            for ij in active_pairs:
                ax0.annotate(f"{ij}", (ij[0], ij[1]), textcoords="offset points", xytext=(3, 3), fontsize=8)
        ax0.set_xlabel("i (basis index)")
        ax0.set_ylabel("j (basis index)")
        ax0.grid(True, alpha=0.25)

        # Right: Top contributions bar chart
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.set_title(f"Top |contribution| terms\n(u_total={u_total:.6e}, d={d:.3e})")
        if len(top) == 0:
            ax1.text(0.5, 0.5, "No contributions selected (check tol/top_k)", ha='center', va='center')
        else:
            labels = [f"{r['inner_idx']}" for r in top]
            vals = [r['contribution'] for r in top]
            ax1.barh(range(len(vals)), vals)
            ax1.set_yticks(range(len(vals)))
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.invert_yaxis()
            ax1.axvline(0.0, color='k', lw=0.8)
            ax1.grid(True, axis='x', alpha=0.25)
            ax1.set_xlabel("u_i · (d · B_i^e)")

        fig.suptitle(
            f"WEB reconstruction at (x={x:.6g}, y={y:.6g}) in domain context | use_norm={use_norm}",
            y=1.00,
        )
        plt.tight_layout()
        plt.show()

        # Print a compact textual breakdown
        print("=" * 70)
        print(f"Point (x,y)=({x:.8g},{y:.8g})  d={d:.8g}  g={g:.8g}  u_total={u_total:.8g}")
        print(f"Active tensor-product indices: {active_pairs}")
        print(f"Active inner: {active_inner_pairs}")
        print(f"Active outer: {active_outer_pairs}")
        print("Top contributions:")
        for r in top[: min(len(top), 10)]:
            w_info = f" w_i={r['w_i']:.3e}" if (use_norm and r.get('w_i') is not None) else ""
            print(
                f"  i={r['inner_idx']}  u_i={r['u_i']:+.3e}  Be={r['Be']:+.3e}{w_info}  contrib={r['contribution']:+.3e}"
            )
        print("=" * 70)

    return details


def visualizeBasisFunctionSupportsInPhysicalDomain(
    model,
    x: float,
    y: float,
    u_reduced,
    p: int,
    q: int,
    knotvector_x,
    knotvector_y,
    bspline_classification,
    extended_basis,
    *,
    show_original_bases: bool = False,
    top_k: int = 12,
):
    """Visualize basis function supports in physical domain with extension coefficients.
    
    Shows:
      - Physical domain with boundary (from distance function)
      - Support regions of active basis functions colored by contribution magnitude
      - Extension coefficients e_{j,i} for outer basis functions in a table
      - The evaluation point marked on the domain
    
    Parameters:
    -----------
    model : SDF model
        Geometry representation
    x, y : float
        Evaluation point in physical coordinates
    u_reduced : array
        WEB coefficients (inner DOFs only)
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array
        Knot vectors
    bspline_classification : dict
        Classification output with 'inner', 'outer', etc.
    extended_basis : dict
        Extended basis coefficients e_{j,i}
    show_original_bases : bool
        If True, show original (non-extended) basis supports separately
    top_k : int
        Show top-k basis functions by contribution magnitude
    """
    # Evaluate at point to get active bases and contributions
    details = visualizeReconstructionAtPointWEB(
        model, x, y, u_reduced, p, q, knotvector_x, knotvector_y,
        bspline_classification, extended_basis, top_k=top_k, show=False
    )
    
    # Unpack details
    active_pairs = details['active_pairs']
    active_inner_pairs = details['active_inner_pairs']
    active_outer_pairs = details['active_outer_pairs']
    top_contributions = details['top_contributions']
    use_norm = details['use_norm']
    
    if details['outside']:
        print(f"Point ({x}, {y}) is outside the domain (d < 0). No visualization.")
        return
    
    inner_set = set(map(tuple, bspline_classification.get('inner', [])))
    outer_set = set(map(tuple, bspline_classification.get('outer', [])))
    inner_bsplines = bspline_classification.get('inner', [])
    
    # --- Helper: compute support region of a basis function in parametric space ---
    def _get_parametric_support(basis_idx, deg, knotvector):
        """Get parametric span [u_min, u_max] of basis function i with degree deg."""
        i = basis_idx
        # B-spline i has support [t_{i}, t_{i+deg+1})
        if i >= len(knotvector) - deg - 1:
            return None
        u_min = float(knotvector[i])
        u_max = float(knotvector[i + deg + 1])
        return u_min, u_max
    
    # --- Helper: evaluate basis function on a grid to find support region ---
    def _evaluate_basis_on_grid(basis_idx, deg, ktvec, axis='x', n_grid=100):
        """Evaluate a 1D basis function on a grid.
        
        Returns (param_vals, basis_vals) where basis_vals > 0 indicates support.
        """
        sup = _get_parametric_support(basis_idx, deg, ktvec)
        if sup is None:
            return np.array([]), np.array([])
        
        u_min, u_max = sup
        # Extend slightly beyond support to show edges clearly
        extend = 0.05 * (u_max - u_min)
        u_vals = np.linspace(max(u_min - extend, float(ktvec[0])), 
                             min(u_max + extend, float(ktvec[-1])), n_grid)
        
        Bsp = FEM._get_bspline_objects(ktvec if axis == 'x' else np.array([0]), 
                                       ktvec if axis == 'y' else np.array([0]), 
                                       deg, deg if axis == 'y' else 0)[0 if axis == 'x' else 1]
        b_vals = np.asarray(Bsp(u_vals), dtype=np.float64)
        return u_vals, b_vals
    
    # --- Create figure with multiple panels ---
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.1, 1.2])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax_phys = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    
    # Panel 1: Parametric support regions
    ax1.set_title(f"Basis function supports in parametric domain\n(Top {top_k} by contribution)")
    
    # Draw support regions for top contributions in parametric (u,v) space
    cmap = plt.cm.RdYlBu_r
    max_contrib = max([abs(c['contribution']) for c in top_contributions] + [1e-10])
    
    # Also plot the domain extent in parametric space
    ax1.axhline(float(knotvector_y[0]), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(float(knotvector_y[-1]), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(float(knotvector_x[0]), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(float(knotvector_x[-1]), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    for idx, contrib_info in enumerate(top_contributions):
        inner_idx = contrib_info['inner_idx']
        contribution = contrib_info['contribution']
        
        # Map contribution magnitude to color
        norm_contrib = np.clip(abs(contribution) / max_contrib, 0, 1)
        color = cmap(norm_contrib)
        
        # Get support in parametric space
        sup_x = _get_parametric_support(inner_idx[0], p, knotvector_x)
        sup_y = _get_parametric_support(inner_idx[1], q, knotvector_y)
        
        if sup_x is not None and sup_y is not None:
            u_min, u_max = sup_x
            v_min, v_max = sup_y
            # Draw rectangle in parametric space
            rect = plt.Rectangle((u_min, v_min), u_max - u_min, v_max - v_min,
                                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
                                label=f"i={inner_idx} (c={contribution:.2e})")
            ax1.add_patch(rect)
    
    # Mark the intersection of active basis supports in parametric space
    # This approximately shows where the point (x,y) maps to in parametric coordinates
    if len(active_inner_pairs) > 0:
        u_min_all = float(knotvector_x[0])
        u_max_all = float(knotvector_x[-1])
        v_min_all = float(knotvector_y[0])
        v_max_all = float(knotvector_y[-1])
        
        for i_idx, j_idx in active_inner_pairs:
            sup_x = _get_parametric_support(i_idx, p, knotvector_x)
            sup_y = _get_parametric_support(j_idx, q, knotvector_y)
            if sup_x and sup_y:
                u_min_all = max(u_min_all, sup_x[0])
                u_max_all = min(u_max_all, sup_x[1])
                v_min_all = max(v_min_all, sup_y[0])
                v_max_all = min(v_max_all, sup_y[1])
        
        
    # Mark the overlap region with a star
    ax1.scatter([x], [y], s=10, marker='+', c='red', edgecolors='darkred', 
                    linewidths=1, label=f'Point ({x:.3g}, {y:.3g}) in parametric space', zorder=15)
    
    ax1.set_xlabel("u (parametric x)")
    ax1.set_ylabel("v (parametric y)")
    ax1.legend(loc='best', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal', adjustable='box')
    
    # Panel 2: Physical domain with boundary contour
    ax_phys.set_title("Physical domain with boundary and basis grid")
    
    # Evaluate distance function on a grid
    marg = 0.1
    x_grid = np.linspace(FEM.DOMAIN["x1"] - marg, FEM.DOMAIN["x2"] + marg, 80)
    y_grid = np.linspace(FEM.DOMAIN["y1"] - marg, FEM.DOMAIN["y2"] + marg, 80)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    D_grid = np.zeros_like(X_grid)
    
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            d_val = model(torch.tensor([[x_grid[i], y_grid[j]]], dtype=TORCH_DTYPE)).item()
            D_grid[j, i] = d_val
    
    # Plot distance field as background
    im_phys = ax_phys.contourf(X_grid, Y_grid, D_grid, levels=15, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(im_phys, ax=ax_phys, label='d(x,y)')
    
    # Overlay boundary (d=0)
    cs = ax_phys.contour(X_grid, Y_grid, D_grid, levels=[0.0], colors='black', linewidths=2)
    ax_phys.clabel(cs, inline=True, fontsize=8)
    
    # Overlay basis function grid (knot lines in parametric space)
    # Get unique knot values (remove duplicates from clamped knots)
    unique_knots_x = np.unique(np.asarray(knotvector_x))
    unique_knots_y = np.unique(np.asarray(knotvector_y))
    
    # Draw vertical lines for x knots
    for u_knot in unique_knots_x:
        ax_phys.axvline(float(u_knot), color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Draw horizontal lines for y knots
    for v_knot in unique_knots_y:
        ax_phys.axhline(float(v_knot), color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Mark evaluation point
    ax_phys.scatter([x], [y], s=400, marker='*', c='red', edgecolors='white', linewidths=2.5, 
                    label=f'Point ({x:.3g}, {y:.3g})', zorder=15)
    ax_phys.scatter([x], [y], s=80, marker='+', c='darkred', linewidths=2.5, zorder=16)
    
    # Mark active basis locations in physical space (sample points)
    for pair in active_inner_pairs[:5]:  # Show a few for clarity
        # Map parametric index to approximate physical location
        i, j = pair
        sup_x = _get_parametric_support(i, p, knotvector_x)
        sup_y = _get_parametric_support(j, q, knotvector_y)
        if sup_x and sup_y:
            u_mid = 0.5 * (sup_x[0] + sup_x[1])
            v_mid = 0.5 * (sup_y[0] + sup_y[1])
            ax_phys.scatter([u_mid], [v_mid], s=80, marker='o', c='blue', alpha=0.6, 
                           edgecolors='darkblue', linewidths=1)
    
    ax_phys.set_xlabel("x")
    ax_phys.set_ylabel("y")
    ax_phys.legend(loc='best', fontsize=8)
    ax_phys.grid(True, alpha=0.2)
    ax_phys.set_aspect('equal', adjustable='box')
    
    # Panel 3: Extension coefficients table
    ax2.axis('tight')
    ax2.axis('off')
    ax2.set_title("Extension coefficients e_{j,i} for active bases", pad=20)
    
    # Build table data: show extension coefficients for active inner bases from active outer bases
    table_data = []
    table_data.append(['Inner i', 'Outer j', 'e_{j,i}', 'Contribution'])
    
    for contrib_info in top_contributions[:top_k]:
        inner_idx = tuple(contrib_info['inner_idx'])
        contrib = contrib_info['contribution']
        basis_coeffs = extended_basis.get(inner_idx, {})
        
        # Show coefficients from active outer bases
        for outer_idx in active_outer_pairs:
            e_ji = basis_coeffs.get(tuple(outer_idx))
            if e_ji is not None and abs(e_ji) > 1e-15:
                table_data.append([
                    str(inner_idx),
                    str(tuple(outer_idx)),
                    f"{e_ji:.4e}",
                    f"{contrib:.4e}"
                ])
    
    if len(table_data) > 1:
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        # Header row styling
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
    else:
        ax2.text(0.5, 0.5, "No extension coefficients found", ha='center', va='center', fontsize=10, transform=ax2.transAxes)
    
    # Panel 3: Contribution magnitudes
    ax3.set_title(f"Top {len(top_contributions)} contributions by magnitude")
    
    labels = [str(c['inner_idx']) for c in top_contributions]
    contribs = [c['contribution'] for c in top_contributions]
    colors_bar = [cmap(np.clip(abs(c) / max_contrib, 0, 1)) for c in contribs]
    
    bars = ax3.barh(range(len(contribs)), contribs, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(contribs)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.invert_yaxis()
    ax3.axvline(0.0, color='k', lw=1)
    ax3.set_xlabel("Contribution u_i · (d · B_i^e)")
    ax3.grid(True, axis='x', alpha=0.25)
    
    fig.suptitle(
        f"WEB Basis Function Supports at (x={x:.6g}, y={y:.6g}) | Active inner: {len(active_inner_pairs)}, outer: {len(active_outer_pairs)}",
        fontsize=12, weight='bold', y=0.98
    )
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.05, wspace=0.35)
    plt.show()
    
    # Print text summary
    print("=" * 80)
    print(f"Basis Support Analysis at (x={x:.8g}, y={y:.8g})")
    print(f"Active tensor-product indices: {len(active_pairs)}")
    print(f"  Inner: {active_inner_pairs}")
    print(f"  Outer: {active_outer_pairs}")
    print(f"\nTop {len(top_contributions)} contributions:")
    for contrib_info in top_contributions:
        inner_idx = tuple(contrib_info['inner_idx'])
        basis_coeffs = extended_basis.get(inner_idx, {})
        print(f"\n  Inner index i={inner_idx}:")
        print(f"    u_i = {contrib_info['u_i']:.6e}")
        print(f"    B_i^e = {contrib_info['Be']:.6e}")
        print(f"    Contribution = {contrib_info['contribution']:.6e}")
        
        # Show extension coefficients
        ext_coeffs_active = {j: basis_coeffs.get(tuple(j)) for j in active_outer_pairs 
                             if basis_coeffs.get(tuple(j)) is not None}
        if ext_coeffs_active:
            print(f"    Extension coefficients e_{{j,i}} from active outer bases:")
            for j, e_val in ext_coeffs_active.items():
                if abs(e_val) > 1e-15:
                    print(f"      e_{{{j},i}} = {e_val:.6e}")
    print("=" * 80)


def evaluateAccuracyWEB(model, u_reduced, p, q, knotvector_x, knotvector_y,
                        bspline_classification, extended_basis, N=1000, seed=None):
    """
    Evaluate the accuracy of WEB-spline solution compared to analytical solution.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model representing the geometry (SDF)
    u_reduced : array-like
        WEB-spline coefficients (inner DOFs only)
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array-like
        Knot vectors
    bspline_classification : dict
        Output from classifyBsplines()
    extended_basis : dict
        Output from buildExtendedBasis()
    N : int
        Number of random evaluation points
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing all error metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points in the domain
    x_samples = np.random.uniform(FEM.DOMAIN["x1"], FEM.DOMAIN["x2"], N)
    y_samples = np.random.uniform(FEM.DOMAIN["y1"], FEM.DOMAIN["y2"], N)
    
    # Lists to store results for valid points
    numerical_vals = []
    analytical_vals = []
    numerical_dx = []
    numerical_dy = []
    analytical_dx = []
    analytical_dy = []
    
    # Evaluate at each point
    points_tensor = torch.tensor(np.column_stack([x_samples, y_samples]), dtype=TORCH_DTYPE)
    
    with torch.no_grad():
        distances = model(points_tensor).numpy().flatten()
    
    for idx in range(N):
        xx, yy = x_samples[idx], y_samples[idx]
        d = distances[idx]
        
        # Skip points outside the domain
        if d < 0:
            continue
        
        # Numerical solution from WEB-splines
        u_h = FEM_WEB.reconstructSolution(xx, yy, u_reduced, model, p, q,
                                          knotvector_x, knotvector_y,
                                          bspline_classification, extended_basis)
        
        du_h_dx, du_h_dy = FEM_WEB.reconstructSolutionGradient(xx, yy, u_reduced, model, p, q,
                                                               knotvector_x, knotvector_y,
                                                               bspline_classification, extended_basis)
        
        # Analytical solution
        u_exact = FEM.solution_function(xx, yy)
        du_exact_dx = FEM.solution_function_derivative_x(xx, yy)
        du_exact_dy = FEM.solution_function_derivative_y(xx, yy)
        
        # Store values
        numerical_vals.append(u_h)
        analytical_vals.append(u_exact)
        numerical_dx.append(du_h_dx)
        numerical_dy.append(du_h_dy)
        analytical_dx.append(du_exact_dx)
        analytical_dy.append(du_exact_dy)
    
    # Convert to numpy arrays
    numerical_vals = np.array(numerical_vals)
    analytical_vals = np.array(analytical_vals)
    numerical_dx = np.array(numerical_dx)
    numerical_dy = np.array(numerical_dy)
    analytical_dx = np.array(analytical_dx)
    analytical_dy = np.array(analytical_dy)
    
    n_valid = len(numerical_vals)
    
    if n_valid == 0:
        return {
            'MSE': np.nan,
            'MAE': np.nan,
            'L_inf': np.nan,
            'relative_error': np.nan,
            'H1_error': np.nan,
            'H1_full': np.nan,
            'n_valid_points': 0
        }
    
    # Compute errors
    error = numerical_vals - analytical_vals
    error_dx = numerical_dx - analytical_dx
    error_dy = numerical_dy - analytical_dy
    
    # MSE
    MSE = np.mean(error ** 2)
    
    # MAE
    MAE = np.mean(np.abs(error))
    
    # L_inf
    L_inf = np.max(np.abs(error))
    
    # Relative error
    MAE_exact = np.mean(np.abs(analytical_vals))
    relative_error = MAE / MAE_exact if MAE_exact > 1e-15 else np.inf
    
    # H1 seminorm
    H1_seminorm_sq = np.sum(error_dx ** 2 + error_dy ** 2)
    H1_error = np.sqrt(H1_seminorm_sq / n_valid)
    
    # Full H1 norm
    H1_full = np.sqrt(np.sum(error ** 2) + H1_seminorm_sq) / np.sqrt(n_valid)
    
    return {
        'MSE': MSE,
        'MAE': MAE,
        'L_inf': L_inf,
        'relative_error': relative_error,
        'H1_error': H1_error,
        'H1_full': H1_full,
        'n_valid_points': n_valid
    }



def computeL2andH1Errors(model, u_reduced, p, q, knotvector_x, knotvector_y,
                         bspline_classification, extended_basis, N=2000, seed=None):
    """
    Compute L2 and H1 errors of WEB-spline solution vs. analytical solution.
    
    Uses the standard definitions:
        L2-error = sqrt(∫ (u - ũ)² d(x,y))
        H1-error = sqrt(∫ (u - ũ)² + (∂/∂x(u - ũ))² + (∂/∂y(u - ũ))² d(x,y))
    
    where u is exact solution and ũ is approximate (WEB) solution.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model representing the geometry (SDF)
    u_reduced : array-like
        WEB-spline coefficients (inner DOFs only)
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array-like
        Knot vectors
    bspline_classification : dict
        Output from classifyBsplines()
    extended_basis : dict
        Output from buildExtendedBasis()
    N : int
        Number of random evaluation points (default 2000)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'L2_error': L2 norm of the error
        - 'H1_error': H1 norm of the error
        - 'H1_seminorm': H1 seminorm (gradient part only)
        - 'n_valid_points': Number of valid points inside domain
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points in the domain
    x_samples = np.random.uniform(FEM.DOMAIN["x1"], FEM.DOMAIN["x2"], N)
    y_samples = np.random.uniform(FEM.DOMAIN["y1"], FEM.DOMAIN["y2"], N)
    
    # Evaluate distances to check which points are inside domain
    points_tensor = torch.tensor(np.column_stack([x_samples, y_samples]), dtype=TORCH_DTYPE)
    
    with torch.no_grad():
        distances = model(points_tensor).numpy().flatten()
    
    # Filter to points inside domain (d >= 0)
    inside_mask = distances >= 0
    x_inside = x_samples[inside_mask]
    y_inside = y_samples[inside_mask]
    n_valid = len(x_inside)
    
    if n_valid == 0:
        print("ERROR: No valid points inside domain!")
        return {
            'L2_error': np.nan,
            'H1_error': np.nan,
            'H1_seminorm': np.nan,
            'n_valid_points': 0
        }
    
    # Accumulate error contributions using Riemann sum approximation
    # The domain Ω has approximate area we need to normalize by
    domain_area = (FEM.DOMAIN["x2"] - FEM.DOMAIN["x1"]) * (FEM.DOMAIN["y2"] - FEM.DOMAIN["y1"])
    
    # Evaluate numerical solution and its derivatives at valid points
    u_numerical = np.zeros(n_valid)
    du_numerical_dx = np.zeros(n_valid)
    du_numerical_dy = np.zeros(n_valid)
    
    for idx in range(n_valid):
        xx, yy = x_inside[idx], y_inside[idx]
        
        u_numerical[idx] = FEM_WEB.reconstructSolution(
            xx, yy, u_reduced, model, p, q, knotvector_x, knotvector_y,
            bspline_classification, extended_basis
        )
        
        du_dx, du_dy = FEM_WEB.reconstructSolutionGradient(
            xx, yy, u_reduced, model, p, q, knotvector_x, knotvector_y,
            bspline_classification, extended_basis
        )
        du_numerical_dx[idx] = du_dx
        du_numerical_dy[idx] = du_dy
    
    # Evaluate exact solution and its derivatives at valid points
    u_exact = np.array([FEM.solution_function(x, y) for x, y in zip(x_inside, y_inside)])
    du_exact_dx = np.array([FEM.solution_function_derivative_x(x, y) for x, y in zip(x_inside, y_inside)])
    du_exact_dy = np.array([FEM.solution_function_derivative_y(x, y) for x, y in zip(x_inside, y_inside)])
    
    # Compute errors
    error_u = u_numerical - u_exact
    error_du_dx = du_numerical_dx - du_exact_dx
    error_du_dy = du_numerical_dy - du_exact_dy
    
    # Compute L2 error: sqrt(mean((u - u_exact)^2)) * sqrt(domain_area)
    L2_error_squared = np.mean(error_u ** 2) * domain_area
    L2_error = np.sqrt(L2_error_squared)
    
    # Compute H1 error: sqrt(mean((u - u_exact)^2 + (du/dx)^2 + (du/dy)^2)) * sqrt(domain_area)
    H1_integrand = error_u ** 2 + error_du_dx ** 2 + error_du_dy ** 2
    H1_error_squared = np.mean(H1_integrand) * domain_area
    H1_error = np.sqrt(H1_error_squared)
    
    # Compute H1 seminorm (gradient part only)
    H1_seminorm_integrand = error_du_dx ** 2 + error_du_dy ** 2
    H1_seminorm_squared = np.mean(H1_seminorm_integrand) * domain_area
    H1_seminorm = np.sqrt(H1_seminorm_squared)
    
    return {
        'L2_error': L2_error,
        'H1_error': H1_error,
        'H1_seminorm': H1_seminorm,
        'n_valid_points': n_valid
    }


def printL2andH1Errors(metrics):
    """Pretty print the L2 and H1 error metrics."""
    print("=" * 60)
    print("WEB-Splines Accuracy: L2 and H1 Error Norms")
    print("=" * 60)
    print(f"  Evaluation points inside domain: {metrics['n_valid_points']}")
    print()
    print(f"  L2-error   = ‖u - ũ‖_L2    = {metrics['L2_error']:.6e}")
    print(f"  H1-error   = ‖u - ũ‖_H1    = {metrics['H1_error']:.6e}")
    print(f"  H1-seminorm= |u - ũ|_H1    = {metrics['H1_seminorm']:.6e}")
    print("=" * 60)


def printErrorMetricsWEB(metrics):
    """Pretty print the error metrics."""
    print("=" * 50)
    print("WEB-Splines Error Metrics Summary")
    print("=" * 50)
    print(f"  Valid evaluation points: {metrics['n_valid_points']}")
    print(f"  MSE (Mean Squared Error):    {metrics['MSE']:.6e}")
    print(f"  MAE (Mean Absolute Error):   {metrics['MAE']:.6e}")
    print(f"  L_inf (Max Absolute Error):  {metrics['L_inf']:.6e}")
    print(f"  Relative Absolute Error:     {metrics['relative_error']:.6e}")
    print(f"  H1 Seminorm Error:           {metrics['H1_error']:.6e}")
    print(f"  H1 Full Norm Error:          {metrics['H1_full']:.6e}")
    print("=" * 50)


def plotErrorHeatmapWEB(model, u_reduced, knotvector_x, knotvector_y, p, q,
                        bspline_classification, extended_basis, N=100):
    """Plot error heatmap for WEB-spline solution."""
    marg = 0.1
    x = np.linspace(FEM.DOMAIN["x1"] - marg, FEM.DOMAIN["x2"] + marg, N)
    y = np.linspace(FEM.DOMAIN["y1"] - marg, FEM.DOMAIN["y2"] + marg, N)
    X, Y = np.meshgrid(x, y)
    
    Z_N = np.zeros((N, N))
    Z_A = np.zeros((N, N))
    
    for idxx, xx in enumerate(x):
        for idxy, yy in enumerate(y):
            d = mesh.distanceFromContur(xx, yy, model)
            if hasattr(d, 'item'):
                d = d.item()
            
            if d >= 0:
                Z_N[idxy, idxx] = FEM_WEB.reconstructSolution(
                    xx, yy, u_reduced, model, p, q,
                    knotvector_x, knotvector_y,
                    bspline_classification, extended_basis
                )
                Z_A[idxy, idxx] = FEM.solution_function(xx, yy)
    
    L1_error = np.abs(Z_N - Z_A)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, L1_error, levels=20, cmap='hot')
    plt.colorbar(label='|u_h - u_exact|')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('WEB-Splines Error Heatmap')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plotSolutionHeatmapWEB(model, u_reduced, knotvector_x, knotvector_y, p, q,
                           bspline_classification, extended_basis, N=100):
    """Plot solution heatmap for WEB-spline solution."""
    marg = 0.05
    x = np.linspace(FEM.DOMAIN["x1"] - marg, FEM.DOMAIN["x2"] + marg, N)
    y = np.linspace(FEM.DOMAIN["y1"] - marg, FEM.DOMAIN["y2"] + marg, N)
    X, Y = np.meshgrid(x, y)
    
    Z_N = np.zeros((N, N))
    Z_A = np.zeros((N, N))
    
    for idxx, xx in enumerate(x):
        for idxy, yy in enumerate(y):
            d = mesh.distanceFromContur(xx, yy, model)
            if hasattr(d, 'item'):
                d = d.item()
            
            if d >= 0:
                Z_N[idxy, idxx] = FEM_WEB.reconstructSolution(
                    xx, yy, u_reduced, model, p, q,
                    knotvector_x, knotvector_y,
                    bspline_classification, extended_basis
                )
                Z_A[idxy, idxx] = FEM.solution_function(xx, yy)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Numerical solution
    im0 = axes[0].contourf(X, Y, Z_N, levels=20)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title('WEB-Spline Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].axis('equal')
    
    # Analytical solution
    im1 = axes[1].contourf(X, Y, Z_A, levels=20)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].axis('equal')
    
    # Error
    im2 = axes[2].contourf(X, Y, np.abs(Z_N - Z_A), levels=20)
    plt.colorbar(im2, ax=axes[2])
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.show()


def compareConditionNumbers(
    model,
    p,
    q,
    knotvector_x,
    knotvector_y,
    xDivision,
    yDivision,
    *,
    extension_strict: bool = True,
    web_use_weight_normalization: bool = False,
    web_ref_weight_eps: float = 1e-10,
):
    """
    Compare condition numbers between standard weighted B-splines and WEB-splines.
    
    This demonstrates the improved conditioning of WEB-splines.
    """
    import FEM as FEM_standard
    
    print("\n" + "=" * 60)
    print("CONDITION NUMBER COMPARISON")
    print("=" * 60)
    
    # Standard weighted B-splines
    print("\n[Standard Weighted B-splines]")
    n_dof_standard = (xDivision + p + 1) * (yDivision + q + 1)
    K_std = np.zeros((n_dof_standard, n_dof_standard))
    F_std = np.zeros(n_dof_standard)
    
    K_std, F_std, etype_std = FEM_standard.processAllElements(
        model, p, q, knotvector_x, knotvector_y, xDivision, yDivision, K_std, F_std
    )
    
    # Get non-zero part for condition number
    non_zero_rows = np.any(K_std != 0, axis=1)
    K_std_reduced = K_std[non_zero_rows][:, non_zero_rows]
    
    try:
        cond_std = np.linalg.cond(K_std_reduced)
        print(f"  DOFs (active): {np.sum(non_zero_rows)}")
        print(f"  Condition number: {cond_std:.2e}")
    except:
        print("  Could not compute condition number")
        cond_std = np.inf
    
    # WEB-splines
    print("\n[WEB-Splines]")
    K_web, F_web, etype_web, bsp_class, ext_basis = FEM_WEB.processAllElementsWEB(
        model,
        p,
        q,
        knotvector_x,
        knotvector_y,
        xDivision,
        yDivision,
        extension_strict=extension_strict,
        web_use_weight_normalization=web_use_weight_normalization,
        web_ref_weight_eps=web_ref_weight_eps,
    )
    
    # Get non-zero part
    non_zero_rows_web = np.any(K_web != 0, axis=1)
    K_web_reduced = K_web[non_zero_rows_web][:, non_zero_rows_web]
    
    try:
        cond_web = np.linalg.cond(K_web_reduced)
        print(f"  DOFs (active): {np.sum(non_zero_rows_web)}")
        print(f"  Condition number: {cond_web:.2e}")
    except:
        print("  Could not compute condition number")
        cond_web = np.inf
    
    # Comparison
    print("\n[Comparison]")
    if cond_std < np.inf and cond_web < np.inf:
        improvement = cond_std / cond_web
        print(f"  Condition number improvement: {improvement:.2f}x")
    
    return {
        'cond_standard': cond_std,
        'cond_web': cond_web,
        'dof_standard': np.sum(non_zero_rows),
        'dof_web': np.sum(non_zero_rows_web)
    }


if __name__ == "__main__":
    import time
    from Geomertry import AnaliticalDistanceCircle
    
    print("\n" + "=" * 60)
    print("WEB-SPLINES EVALUATION DEMO")
    print("=" * 60)
    
    model = AnaliticalDistanceCircle()
    
    DIVISIONS = 10
    ORDER = 1
    DELTA = 0.005
    
    default = mesh.getDefaultValues(div=DIVISIONS, order=ORDER, delta=DELTA)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knotvector_u, knotvector_w, weights, ctrlpts = mesh.generateRectangularMesh(*default)
    
    # Assemble and solve with WEB-splines
    EXTENSION_STRICT = True
    WEB_USE_WEIGHT_NORMALIZATION = False
    K, F, etype, bsp_class, ext_basis = FEM_WEB.processAllElementsWEB(
        model,
        p,
        q,
        knotvector_u,
        knotvector_w,
        xDivision,
        yDivision,
        extension_strict=EXTENSION_STRICT,
        web_use_weight_normalization=WEB_USE_WEIGHT_NORMALIZATION,
    )
    
    u = FEM_WEB.solveWEB(K, F)
    
    # Evaluate accuracy
    metrics = evaluateAccuracyWEB(model, u, p, q, knotvector_u, knotvector_w,
                                  bsp_class, ext_basis, N=5000, seed=42)
    printErrorMetricsWEB(metrics)
    
    # Compare condition numbers
    compareConditionNumbers(
        model,
        p,
        q,
        knotvector_u,
        knotvector_w,
        xDivision,
        yDivision,
        extension_strict=EXTENSION_STRICT,
        web_use_weight_normalization=WEB_USE_WEIGHT_NORMALIZATION,
    )
