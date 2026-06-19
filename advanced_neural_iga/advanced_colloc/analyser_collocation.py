"""
Analyser for WEB-Spline Collocation method.

Similar to analyser.py but uses the collocation_WEB solver instead of FEM.

Author: Neural IGA Research
Date: 2026-01-26
"""

import numpy as np
import torch
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)

import collocation_WEB as cWEB
import Geomertry
from network_defs import load_test_model


def save_simulation_results(filename: str, mdl_name: str, H_values: List[int], 
                            n_values: List[int], all_eval_stats: Dict,
                            function_case: int = None, domain: Dict = None):
    """
    Save simulation results to a JSON file.
    
    Parameters:
    -----------
    filename : str
        Path to the output file
    mdl_name : str
        Name of the model used
    H_values : list
        List of grid resolution values tested
    n_values : list
        List of B-spline degrees tested
    all_eval_stats : dict
        Dictionary with keys as (n, H) tuples and values as eval_stats dicts
    function_case : int, optional
        The FUNCTION_CASE parameter
    domain : dict, optional
        The physical domain bounds
    """
    # Convert tuple keys to string keys for JSON compatibility
    results_serializable = {}
    for (n, H), stats in all_eval_stats.items():
        key = f"degree_{n}_H_{H}"
        # Convert numpy types to Python native types
        stats_clean = {}
        for k, v in stats.items():
            if isinstance(v, (np.floating, float)):
                stats_clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                stats_clean[k] = int(v)
            elif isinstance(v, dict):
                # Handle nested dicts like rtimes
                stats_clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                  for kk, vv in v.items()}
            else:
                stats_clean[k] = v
        results_serializable[key] = stats_clean
    
    data = {
        "method": "WEB-Spline Collocation",
        "model_name": mdl_name,
        "H_values": H_values,
        "n_values": n_values,
        "function_case": function_case,
        "domain": domain,
        "timestamp": datetime.now().isoformat(),
        "eval_stats": results_serializable
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filename}")


def run_collocation_analysis(
    model: torch.nn.Module,
    mdl_name: str,
    H_values: List[int],
    n_values: List[int],
    domain: Dict[str, float],
    function_case: int = 1,
    verbose: bool = True,
    save_results: bool = True,
) -> Dict:
    """
    Run convergence analysis using WEB-spline collocation.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The weight function model (SDF)
    mdl_name : str
        Name of the model for saving
    H_values : list
        List of grid resolutions to test
    n_values : list
        List of B-spline degrees to test
    domain : dict
        Physical domain {'x1': ..., 'x2': ..., 'y1': ..., 'y2': ...}
    function_case : int
        Test function case (1-7)
    verbose : bool
        Print progress
    save_results : bool
        Save results to JSON file
    
    Returns:
    --------
    all_eval_stats : dict
        Dictionary with all evaluation statistics
    """
    # Set up global function case
    cWEB.FUNCTION_CASE = function_case
    
    # Create weight function with domain
    wfct = cWEB.NeuralWeightFunction(model=model, domain=domain)
    
    # Create domain transformer for exact solution evaluation
    transformer = cWEB.create_domain_transformer(domain)
    u_exact_transformed = transformer.wrap_function(cWEB.solution_function)
    du_dx_transformed = transformer.wrap_derivative_x(cWEB.solution_function_derivative_x)
    du_dy_transformed = transformer.wrap_derivative_y(cWEB.solution_function_derivative_y)
    
    all_eval_stats = {}
    
    for n in n_values:
        # Precompute collocation data for this degree
        CD = cWEB.compute_collocation_data(n, J_MAX=16)
        
        for H in H_values:
            if verbose:
                print("=" * 60)
                print(f"Running: degree n={n}, resolution H={H}")
                print("=" * 60)
            
            try:
                # Solve
                Uxy, xB, yB, con, dim_sys, rtimes = cWEB.collocation_2d(
                    n=n,
                    H=H,
                    wfct=wfct,
                    f=cWEB.load_function,
                    CD=CD,
                    verbose=verbose,
                    domain=domain
                )
                
                if Uxy is None:
                    print(f"  Solver failed for n={n}, H={H}")
                    all_eval_stats[(n, H)] = {"error": "solver_failed"}
                    continue
                
                # Compute errors
                w_grid, _, _, _, _ = wfct(xB, yB)
                mask = w_grid > 0
                
                uxy_exact = np.zeros_like(Uxy)
                uxy_exact[mask] = u_exact_transformed(xB[mask], yB[mask])
                
                error = Uxy[mask] - uxy_exact[mask]
                exact_vals = uxy_exact[mask]
                
                # Error metrics
                max_exact = np.max(np.abs(exact_vals)) if np.any(exact_vals != 0) else 1.0
                l2_exact = np.sqrt(np.sum(exact_vals ** 2)) if np.any(exact_vals != 0) else 1.0
                
                ErrMax = np.max(np.abs(error)) / max_exact
                ErrL2 = np.sqrt(np.sum(error ** 2)) / l2_exact
                Err_MAE = np.mean(np.abs(error))
                Err_L_inf = np.max(np.abs(error))
                
                # H1 semi-norm error
                h = 1.0 / H
                dU_dx_num, dU_dy_num = cWEB.compute_numerical_gradient(Uxy, h)
                
                dU_dx_ex = np.zeros_like(Uxy)
                dU_dy_ex = np.zeros_like(Uxy)
                dU_dx_ex[mask] = du_dx_transformed(xB[mask], yB[mask])
                dU_dy_ex[mask] = du_dy_transformed(xB[mask], yB[mask])
                
                grad_error_x = dU_dx_num[mask] - dU_dx_ex[mask]
                grad_error_y = dU_dy_num[mask] - dU_dy_ex[mask]
                grad_exact_norm = np.sqrt(np.sum(dU_dx_ex[mask] ** 2 + dU_dy_ex[mask] ** 2))
                
                if grad_exact_norm > 0:
                    H1_semi = np.sqrt(np.sum(grad_error_x ** 2 + grad_error_y ** 2)) / grad_exact_norm
                else:
                    H1_semi = np.nan
                
                H1_error = np.sqrt(ErrL2 ** 2 + H1_semi ** 2) if not np.isnan(H1_semi) else np.nan
                
                # Store results
                eval_stats = {
                    "n": n,
                    "H": H,
                    "h": h,
                    "ErrMax": ErrMax,
                    "ErrL2": ErrL2,
                    "MAE": Err_MAE,
                    "L_inf": Err_L_inf,
                    "H1_semi": H1_semi if not np.isnan(H1_semi) else None,
                    "H1_error": H1_error if not np.isnan(H1_error) else None,
                    "condition": con if not np.isnan(con) else None,
                    "dim_sys": dim_sys,
                    "n_interior_points": int(np.sum(mask)),
                    "rtimes": rtimes,
                }
                
                all_eval_stats[(n, H)] = eval_stats
                
                if verbose:
                    print(f"\nResults for n={n}, H={H}:")
                    print(f"  Relative L2 error:  {ErrL2:.6e}")
                    print(f"  Relative max error: {ErrMax:.6e}")
                    print(f"  MAE:                {Err_MAE:.6e}")
                    print(f"  H1 semi-norm:       {H1_semi:.6e}" if not np.isnan(H1_semi) else "  H1 semi-norm: N/A")
                    print(f"  System dimension:   {dim_sys}")
                    print(f"  Total time:         {rtimes['total']:.3f}s")
                    
            except Exception as e:
                print(f"  Error for n={n}, H={H}: {e}")
                all_eval_stats[(n, H)] = {"error": str(e)}
    
    # Print convergence summary
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"{'n':>3} {'H':>6} {'h':>10} {'ErrL2':>12} {'ErrMax':>12} {'H1_semi':>12} {'dim':>8} {'time':>8}")
    print("-" * 80)
    
    for n in n_values:
        for H in H_values:
            stats = all_eval_stats.get((n, H), {})
            if "error" in stats:
                print(f"{n:>3} {H:>6} {'':>10} {'FAILED':>12}")
            else:
                h = stats.get('h', np.nan)
                ErrL2 = stats.get('ErrL2', np.nan)
                ErrMax = stats.get('ErrMax', np.nan)
                H1_semi = stats.get('H1_semi', np.nan)
                dim_sys = stats.get('dim_sys', 0)
                total_time = stats.get('rtimes', {}).get('total', np.nan)
                
                print(f"{n:>3} {H:>6} {h:>10.4f} {ErrL2:>12.4e} {ErrMax:>12.4e} "
                      f"{H1_semi if H1_semi else 'N/A':>12} {dim_sys:>8} {total_time:>8.2f}s")
    
    # Compute convergence rates
    print("\n" + "=" * 80)
    print("CONVERGENCE RATES (log2 ratio between consecutive H values)")
    print("=" * 80)
    
    for n in n_values:
        print(f"\nDegree n = {n}:")
        prev_stats = None
        prev_H = None
        for H in sorted(H_values):
            stats = all_eval_stats.get((n, H), {})
            if "error" not in stats and prev_stats is not None:
                ratio_L2 = prev_stats['ErrL2'] / stats['ErrL2']
                rate_L2 = np.log2(ratio_L2) if ratio_L2 > 0 else np.nan
                
                ratio_max = prev_stats['ErrMax'] / stats['ErrMax']
                rate_max = np.log2(ratio_max) if ratio_max > 0 else np.nan
                
                print(f"  H: {prev_H} -> {H}: L2 rate = {rate_L2:.2f}, Max rate = {rate_max:.2f}")
            
            if "error" not in stats:
                prev_stats = stats
                prev_H = H
    
    # Save results
    if save_results:
        filename = f"collocation_results_{mdl_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_simulation_results(
            filename=filename,
            mdl_name=mdl_name,
            H_values=H_values,
            n_values=n_values,
            all_eval_stats=all_eval_stats,
            function_case=function_case,
            domain=domain
        )
    
    return all_eval_stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    # -------------
    
    # Model selection
    mdl_name = "AnalyticalDistanceCircle"
    model = Geomertry.AnaliticalDistanceCircle()
    
    # Alternatively, use a trained neural network:
    # mdl_name = "SIREN_circle_1_2"
    # model = load_test_model(mdl_name, type="SIREN", 
    #                         params={"architecture": [2, 256, 256, 256, 1], 
    #                                 "w_0": 15.0, "w_hidden": 30.0})
    
    # Physical domain (where weight function and test functions are defined)
    DOMAIN = {"x1": -1, "x2": 1, "y1": -1, "y2": 1}
    
    # Test function case
    # 1: u = x*(x²+y²-1), f = -8x (circle domain, homogeneous Dirichlet)
    # 2: u = cos(π/2*(x²+y²))
    # 5: u = sin(2πx)sin(2πy) (L-shape domain)
    FUNCTION_CASE = 1
    
    # Grid resolutions to test
    H_VALUES = [10, 20, 40, 80]
    
    # B-spline degrees to test
    N_VALUES = [2, 3, 4]
    
    # Run analysis
    print("=" * 80)
    print("WEB-SPLINE COLLOCATION CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"Model: {mdl_name}")
    print(f"Domain: {DOMAIN}")
    print(f"Function case: {FUNCTION_CASE}")
    print(f"H values: {H_VALUES}")
    print(f"Degrees: {N_VALUES}")
    print("=" * 80 + "\n")
    
    results = run_collocation_analysis(
        model=model,
        mdl_name=mdl_name,
        H_values=H_VALUES,
        n_values=N_VALUES,
        domain=DOMAIN,
        function_case=FUNCTION_CASE,
        verbose=True,
        save_results=True
    )
