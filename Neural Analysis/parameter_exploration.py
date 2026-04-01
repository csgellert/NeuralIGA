"""
Parameter Exploration Script for Neural IGA Collocation Enhancement

This script systematically tests different combinations of:
- Learning rates
- Network architectures 
- SIREN omega parameters

Goal: Find the best configuration to minimize derivative and second derivative errors.

Run this after executing the main notebook up to the pre-computation of the pool data.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add Neural Analysis to path
NA_DIR = os.path.join(os.getcwd(), "Neural Analysis")
if NA_DIR not in sys.path:
    sys.path.insert(0, NA_DIR)

import collocation_WEB as colloc
import Geomertry as geom
import network_defs as net

# Assumes these are already defined from the main notebook:
# - POOL_SIZE, BATCH_SIZE, EPS_BAND, CUTOFF_SIGMA, DOWNSCALE
# - pool_x, pool_y, pool_colloc_residual, pool_psi, pool_psi_x, pool_psi_y, pool_psi_xx, pool_psi_yy
# - recon_info_corr, wfct_phys_corr, verts_np_corr, VERTS_T, n_edges
# - U_EXACT_TYPE, CORR_N, CORR_H, CORR_GEOM_NAME

def compute_edge_distances(crd, verts_t):
    """Signed halfplane distances: crd -> dists (differentiable w.r.t. crd)."""
    hp = geom._halfplane_distances(crd, verts_t)
    return torch.stack(hp, dim=-1)

def compute_exact_derivatives(wfct, x, y, u_type):
    """Compute exact derivatives for manufactured solution."""
    w, wx, wy, wxx, wyy = wfct(x, y)
    w, wx, wy, wxx, wyy = [np.asarray(a).ravel() for a in (w, wx, wy, wxx, wyy)]
    
    if u_type == "w2":
        ux = 2.0 * w * wx
        uy = 2.0 * w * wy
        uxx = 2.0 * (wx**2 + w * wxx)
        uyy = 2.0 * (wy**2 + w * wyy)
    elif u_type == "expw":
        ew = np.exp(w)
        ux = ew * wx
        uy = ew * wy
        uxx = ew * (wx**2 + wxx)
        uyy = ew * (wy**2 + wyy)
    else:
        raise ValueError(f"Unknown u_type={u_type}")
    
    return ux, uy, uxx, uyy


def train_and_evaluate_config(lr, architecture, omega_first, omega_hidden,
                               pool_data, geometry_data, cutoff_sigma=None,
                               num_epochs=300, verbose=False):
    """
    Train a neural network with given config and return derivative errors.
    Uses direct addition (no psi weighting).

    Parameters:
    -----------
    lr : float
        Learning rate
    architecture : list
        Network architecture [input, hidden..., output]
    omega_first, omega_hidden : float
        SIREN frequency parameters
    pool_data : dict
        Pre-computed pool data
    geometry_data : dict
        Geometry and reconstruction info
    cutoff_sigma : float or None
        Unused (kept for API compatibility)
    num_epochs : int
        Number of training epochs
    verbose : bool
        Print training progress

    Returns:
    --------
    dict with derivative errors and metrics
    """
    # Unpack data
    POOL_SIZE = pool_data['POOL_SIZE']
    BATCH_SIZE = pool_data['BATCH_SIZE']
    DOWNSCALE = pool_data['DOWNSCALE']
    EPS_BAND = pool_data.get('EPS_BAND', 0.04)
    pool_x = pool_data['pool_x']
    pool_y = pool_data['pool_y']
    pool_colloc_residual = pool_data['pool_colloc_residual']

    recon_info_corr = geometry_data['recon_info']
    wfct_phys_corr = geometry_data['wfct_phys']
    verts_np_corr = geometry_data['verts_np']
    VERTS_T = geometry_data['VERTS_T']
    n_edges = geometry_data['n_edges']
    U_EXACT_TYPE = geometry_data['U_EXACT_TYPE']

    # Initialize model and optimizer
    model_test = net.Siren(
        architecture=architecture, outermost_linear=True,
        first_omega_0=omega_first, hidden_omega_0=omega_hidden
    ).double()
    opt_test = torch.optim.Adam(model_test.parameters(), lr=lr)

    # Training loop
    loss_hist = []
    rng_local = np.random.default_rng(123)

    t_start = time.time()
    for epoch in range(num_epochs):
        idx = rng_local.choice(POOL_SIZE, size=BATCH_SIZE, replace=False)

        bx, by = pool_x[idx], pool_y[idx]
        b_colloc_res = pool_colloc_residual[idx]

        colloc_residual_t = torch.tensor(b_colloc_res, dtype=torch.float64)

        crd = torch.tensor(np.stack([bx, by], axis=-1), dtype=torch.float64,
                          requires_grad=True)
        dists = compute_edge_distances(crd, VERTS_T)
        u_nn = model_test(dists).squeeze() * DOWNSCALE

        grad_u_nn = torch.autograd.grad(
            u_nn, crd, grad_outputs=torch.ones_like(u_nn),
            create_graph=True, retain_graph=True
        )[0]
        u_nn_x = grad_u_nn[:, 0]
        u_nn_y = grad_u_nn[:, 1]

        u_nn_xx = torch.autograd.grad(
            u_nn_x, crd, grad_outputs=torch.ones_like(u_nn_x),
            create_graph=True, retain_graph=True
        )[0][:, 0]
        u_nn_yy = torch.autograd.grad(
            u_nn_y, crd, grad_outputs=torch.ones_like(u_nn_y),
            create_graph=True, retain_graph=True
        )[0][:, 1]

        # Direct Laplacian (no psi product rule)
        laplacian_u_nn = u_nn_xx + u_nn_yy

        total_residual = colloc_residual_t + laplacian_u_nn
        loss = (total_residual ** 2).mean()

        opt_test.zero_grad()
        loss.backward()
        opt_test.step()

        loss_hist.append(loss.item())

        if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
            print(f"  epoch {epoch:4d}  loss={loss.item():.6e}")

    t_train = time.time() - t_start

    # Evaluate on grid
    t_eval_start = time.time()
    HN_eval = 80
    xmin_e = float(verts_np_corr[:, 0].min()) - 0.05
    xmax_e = float(verts_np_corr[:, 0].max()) + 0.05
    ymin_e = float(verts_np_corr[:, 1].min()) - 0.05
    ymax_e = float(verts_np_corr[:, 1].max()) + 0.05
    gx_e = np.linspace(xmin_e, xmax_e, HN_eval)
    gy_e = np.linspace(ymin_e, ymax_e, HN_eval)
    GX_e, GY_e = np.meshgrid(gx_e, gy_e)
    flat_xe, flat_ye = GX_e.ravel(), GY_e.ravel()

    # Collocation derivatives
    u_col_e, ux_col_e, uy_col_e, uxx_col_e, uyy_col_e = \
        colloc.reconstruct_collocation_hessian_diag(
            flat_xe, flat_ye, recon_info_corr, wfct_phys_corr)

    # Inside mask
    w_e, *_ = wfct_phys_corr(flat_xe, flat_ye)
    w_e = np.asarray(w_e).ravel()
    inside_e = w_e > 0

    # SDF for band mask
    import Geomertry as geom
    with torch.no_grad():
        crd_sdf_e = torch.tensor(np.stack([flat_xe, flat_ye], axis=-1), dtype=torch.float64)
        sdf_e = geom.convex_polygon_sdf(crd_sdf_e, crd_sdf_e.new_tensor(verts_np_corr)).numpy().ravel()
    band_e = ((sdf_e > 0) & (sdf_e < EPS_BAND)).astype(np.float64)

    # Neural network derivatives
    crd_e = torch.tensor(np.stack([flat_xe, flat_ye], axis=-1),
                        dtype=torch.float64, requires_grad=True)
    dists_e = compute_edge_distances(crd_e, VERTS_T)
    u_nn_e = model_test(dists_e).squeeze() * DOWNSCALE
    grad_u_nn_e = torch.autograd.grad(
        u_nn_e.sum(), crd_e, create_graph=True, retain_graph=True)[0]
    u_nn_x_e = grad_u_nn_e[:, 0]
    u_nn_y_e = grad_u_nn_e[:, 1]
    u_nn_xx_e = torch.autograd.grad(
        u_nn_x_e.sum(), crd_e, create_graph=False, retain_graph=True)[0][:, 0]
    u_nn_yy_e = torch.autograd.grad(
        u_nn_y_e.sum(), crd_e, create_graph=False)[0][:, 1]

    u_nn_x_np_e = u_nn_x_e.detach().numpy()
    u_nn_y_np_e = u_nn_y_e.detach().numpy()
    u_nn_xx_np_e = u_nn_xx_e.detach().numpy()
    u_nn_yy_np_e = u_nn_yy_e.detach().numpy()

    # Corrected derivatives (direct addition within band)
    ux_corr_e = ux_col_e + band_e * u_nn_x_np_e
    uy_corr_e = uy_col_e + band_e * u_nn_y_np_e
    uxx_corr_e = uxx_col_e + band_e * u_nn_xx_np_e
    uyy_corr_e = uyy_col_e + band_e * u_nn_yy_np_e
    
    # Exact derivatives
    ux_ex_e, uy_ex_e, uxx_ex_e, uyy_ex_e = compute_exact_derivatives(
        wfct_phys_corr, flat_xe, flat_ye, U_EXACT_TYPE)
    
    # Errors
    err_ux_col_e = np.abs(ux_col_e - ux_ex_e)
    err_uy_col_e = np.abs(uy_col_e - uy_ex_e)
    err_uxx_col_e = np.abs(uxx_col_e - uxx_ex_e)
    err_uyy_col_e = np.abs(uyy_col_e - uyy_ex_e)
    
    err_ux_corr_e = np.abs(ux_corr_e - ux_ex_e)
    err_uy_corr_e = np.abs(uy_corr_e - uy_ex_e)
    err_uxx_corr_e = np.abs(uxx_corr_e - uxx_ex_e)
    err_uyy_corr_e = np.abs(uyy_corr_e - uyy_ex_e)
    
    grad_err_corr_e = np.sqrt(err_ux_corr_e**2 + err_uy_corr_e**2)
    hess_err_corr_e = np.sqrt(err_uxx_corr_e**2 + err_uyy_corr_e**2)
    
    grad_err_col_e = np.sqrt(err_ux_col_e**2 + err_uy_col_e**2)
    hess_err_col_e = np.sqrt(err_uxx_col_e**2 + err_uyy_col_e**2)
    
    t_eval = time.time() - t_eval_start
    
    # Collect results
    results = {
        'loss_history': loss_hist,
        'final_loss': loss_hist[-1] if loss_hist else float('inf'),
        'train_time': t_train,
        'eval_time': t_eval,
        # Max errors (corrected)
        'u_x_max': float(np.max(err_ux_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_y_max': float(np.max(err_uy_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_xx_max': float(np.max(err_uxx_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_yy_max': float(np.max(err_uyy_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        # Mean errors (corrected)
        'u_x_mean': float(np.mean(err_ux_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_y_mean': float(np.mean(err_uy_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_xx_mean': float(np.mean(err_uxx_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_yy_mean': float(np.mean(err_uyy_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        # Norm errors
        'grad_max': float(np.max(grad_err_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'hess_max': float(np.max(hess_err_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'grad_mean': float(np.mean(grad_err_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        'hess_mean': float(np.mean(hess_err_corr_e[inside_e])) if np.any(inside_e) else float('nan'),
        # Baseline errors (for comparison)
        'u_x_max_base': float(np.max(err_ux_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_y_max_base': float(np.max(err_uy_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_xx_max_base': float(np.max(err_uxx_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_yy_max_base': float(np.max(err_uyy_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_x_mean_base': float(np.mean(err_ux_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_y_mean_base': float(np.mean(err_uy_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_xx_mean_base': float(np.mean(err_uxx_col_e[inside_e])) if np.any(inside_e) else float('nan'),
        'u_yy_mean_base': float(np.mean(err_uyy_col_e[inside_e])) if np.any(inside_e) else float('nan'),
    }
    
    return results


def print_config_results(config_name, results):
    """Pretty print results from a configuration."""
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Final loss: {results['final_loss']:.6e}")
    print(f"Training time: {results['train_time']:.1f}s,  Eval time: {results['eval_time']:.1f}s")
    print(f"\nDerivative Max Errors (Corrected vs Baseline):")
    print(f"  u_x:   {results['u_x_max']:.3e}  (baseline: {results['u_x_max_base']:.3e}, improvement: {results['u_x_max_base']/max(results['u_x_max'],1e-30):.1f}x)")
    print(f"  u_y:   {results['u_y_max']:.3e}  (baseline: {results['u_y_max_base']:.3e}, improvement: {results['u_y_max_base']/max(results['u_y_max'],1e-30):.1f}x)")
    print(f"  u_xx:  {results['u_xx_max']:.3e}  (baseline: {results['u_xx_max_base']:.3e}, improvement: {results['u_xx_max_base']/max(results['u_xx_max'],1e-30):.1f}x)")
    print(f"  u_yy:  {results['u_yy_max']:.3e}  (baseline: {results['u_yy_max_base']:.3e}, improvement: {results['u_yy_max_base']/max(results['u_yy_max'],1e-30):.1f}x)")
    print(f"\nDerivative Mean Errors (Corrected vs Baseline):")
    print(f"  u_x:   {results['u_x_mean']:.3e}  (baseline: {results['u_x_mean_base']:.3e}, improvement: {results['u_x_mean_base']/max(results['u_x_mean'],1e-30):.1f}x)")
    print(f"  u_y:   {results['u_y_mean']:.3e}  (baseline: {results['u_y_mean_base']:.3e}, improvement: {results['u_y_mean_base']/max(results['u_y_mean'],1e-30):.1f}x)")
    print(f"  u_xx:  {results['u_xx_mean']:.3e}  (baseline: {results['u_xx_mean_base']:.3e}, improvement: {results['u_xx_mean_base']/max(results['u_xx_mean'],1e-30):.1f}x)")
    print(f"  u_yy:  {results['u_yy_mean']:.3e}  (baseline: {results['u_yy_mean_base']:.3e}, improvement: {results['u_yy_mean_base']/max(results['u_yy_mean'],1e-30):.1f}x)")
    print(f"\nNorm Errors:")
    print(f"  Gradient max:  {results['grad_max']:.3e},  mean: {results['grad_mean']:.3e}")
    print(f"  Hessian max:   {results['hess_max']:.3e},  mean: {results['hess_mean']:.3e}")


if __name__ == "__main__":
    print("This script should be run from within the Jupyter notebook environment")
    print("after the pool data has been precomputed.")
