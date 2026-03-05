import math
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from scipy import sparse

def runHolligTransformationStudy(
    divisions=20,
    order=2,
    delta_sim=0.005,
    n_samples=2000,
    save_filename=None
):
    """
    Run a comprehensive study of Höllig transformation parameters.
    
    Tests different combinations of delta and gamma parameters for the Höllig
    weight transformation and compares against no transformation.
    
    Parameters:
    -----------
    divisions : int
        Number of mesh divisions (default 20)
    order : int
        B-spline order (default 2)
    delta_sim : float
        Simulation domain extension delta (default 0.005)
    n_samples : int
        Number of sample points for error evaluation (default 2000)
    save_filename : str, optional
        Output JSON filename. If None, generates timestamp-based name.
    
    Returns:
    --------
    dict : Dictionary containing all results
    """
    import FEM
    import FEM_WEB
    import mesh
    import Geomertry
    import evaluation_WEB
    
    # Force CASE=3
    original_case = FEM.FUNCTION_CASE
    FEM.FUNCTION_CASE = 5
    
    print("="*80)
    print("HÖLLIG TRANSFORMATION PARAMETER STUDY")
    print("="*80)
    print(f"Configuration: divisions={divisions}, order={order}, CASE={FEM.FUNCTION_CASE}")
    
    # Element width calculation
    domain_width = FEM.DOMAIN["x2"] - FEM.DOMAIN["x1"]
    element_width = domain_width / divisions
    print(f"Element width: {element_width:.6f}")
    print(f"Simulation delta: {delta_sim:.6f}")
    
    # Model setup
    model = Geomertry.AnaliticalDistanceLshape()
    
    # Define parameter ranges
    # Delta values: fraction of elements that should fall within delta strip
    # Element width ≈ 0.1 for divisions=20, so delta values correspond to multiples of element width
    # BUT: distances near boundary are typically small, so we need delta >> element_width
    delta_multiples = [0.5, 1, 2, 3, 4, 5, 8]  # multiples of element width
    delta_values = [mult * element_width for mult in delta_multiples]
    delta_values.append(1.0)
    
    # Gamma values: controls smoothness of transition
    gamma_values = [1, 2, 3, 4]
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'divisions': divisions,
            'order': order,
            'delta_sim': delta_sim,
            'element_width': element_width,
            'n_samples': n_samples,
            'function_case': FEM.FUNCTION_CASE,
            'geometry': 'AnalyticalDistanceLshape'
        },
        'runs': []
    }
    
    # Test without transformation first
    print("\n" + "="*80)
    print("TEST 1: No transformation (baseline)")
    print("="*80)
    
    original_transform = mesh.TRANSFORM
    mesh.TRANSFORM = None
    
    try:
        result_no_trf = _run_single_simulation(
            model, divisions, order, delta_sim, n_samples
        )
        result_no_trf['transform'] = None
        result_no_trf['delta'] = None
        result_no_trf['gamma'] = None
        results['runs'].append(result_no_trf)
        
        print(f"  L2 error: {result_no_trf['L2_error']:.6e}")
        print(f"  H1 error: {result_no_trf['H1_error']:.6e}")
        print(f"  MAE: {result_no_trf['MAE']:.6e}")
        print(f"  L_inf: {result_no_trf['L_inf']:.6e}")
    except Exception as e:
        print(f"  ERROR: {e}")
        result_no_trf = None
    
    # Test with Höllig transformation for each combination
    mesh.TRANSFORM = "hollig"
    
    total_runs = len(delta_values) * len(gamma_values)
    current_run = 0
    
    for delta in delta_values:
        for gamma in gamma_values:
            current_run += 1
            
            print(f"\n{'='*80}")
            print(f"TEST {current_run + 1}/{total_runs + 1}: Höllig transform")
            print(f"  delta = {delta:.6f} ({delta/element_width:.1f} × element_width)")
            print(f"  gamma = {gamma}")
            print("="*80)
            
            mesh.DELTA_HOLLIG = delta
            mesh.GAMMA_HOLLIG = gamma
            
            try:
                result = _run_single_simulation(
                    model, divisions, order, delta_sim, n_samples
                )
                result['transform'] = 'Hollig'
                result['delta'] = delta
                result['delta_multiple'] = delta / element_width
                result['gamma'] = gamma
                results['runs'].append(result)
                
                print(f"  L2 error: {result['L2_error']:.6e}")
                print(f"  H1 error: {result['H1_error']:.6e}")
                print(f"  MAE: {result['MAE']:.6e}")
                print(f"  L_inf: {result['L_inf']:.6e}")
                
                if result_no_trf:
                    improvement_mae = (result_no_trf['MAE'] - result['MAE']) / result_no_trf['MAE'] * 100
                    improvement_linf = (result_no_trf['L_inf'] - result['L_inf']) / result_no_trf['L_inf'] * 100
                    print(f"  Improvement vs baseline: MAE {improvement_mae:+.2f}%, L_inf {improvement_linf:+.2f}%")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Restore original settings
    mesh.TRANSFORM = original_transform
    FEM.FUNCTION_CASE = original_case
    
    # Save results
    if save_filename is None:
        save_filename = f"hollig_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(save_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {save_filename}")
    print("="*80)
    
    return results


def _run_single_simulation(model, divisions, order, delta_sim, n_samples):
    """Helper function to run a single simulation and collect metrics."""
    import FEM
    import FEM_WEB
    import mesh
    import evaluation_WEB
    
    # Debug: Print current transformation settings
    print(f"    [DEBUG] Current mesh.TRANSFORM: {mesh.TRANSFORM}")
    if mesh.TRANSFORM and mesh.TRANSFORM.lower() == "hollig":
        print(f"    [DEBUG] Current mesh.DELTA_HOLLIG: {mesh.DELTA_HOLLIG}")
        print(f"    [DEBUG] Current mesh.GAMMA_HOLLIG: {mesh.GAMMA_HOLLIG}")
        
        # Test transformation by evaluating at a test point near boundary
        test_point = torch.tensor([0.5, 0.5], dtype=torch.float64)
        raw_d = model(test_point).item()
        transformed_d = mesh.hollig_weight(torch.tensor(raw_d, dtype=torch.float64), 
                                          delta=mesh.DELTA_HOLLIG, 
                                          gamma=mesh.GAMMA_HOLLIG).item()
        print(f"    [DEBUG] Test point [0.5, 0.5]: raw_d={raw_d:.6f}, transformed_d={transformed_d:.6f}")
    
    # Setup mesh
    default = mesh.getDefaultValues(div=divisions, order=order, delta=delta_sim)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knotvector_u, knotvector_w, weights, ctrlpts = mesh.generateRectangularMesh(*default)
    
    # Initialize system
    matrix_size = (xDivision + p + 1) * (yDivision + q + 1)
    K = sparse.lil_matrix((matrix_size, matrix_size), dtype=np.float64)
    F = np.zeros(matrix_size)
    
    # Assembly (transformation applied during quadrature)
    start_time = time.time()
    K, F, etype = FEM.processAllElements(
        model, p, q, knotvector_u, knotvector_w, xDivision, yDivision, K, F
    )
    assembly_time = time.time() - start_time
    
    # Check K matrix characteristics to see if transformation had effect
    K_dense = K.toarray() if hasattr(K, 'toarray') else K
    K_norm = np.linalg.norm(K_dense)
    K_trace = np.trace(K_dense)
    print(f"    [DEBUG] Assembly completed: ||K||={K_norm:.6e}, trace(K)={K_trace:.6e}")
    print(f"    [DEBUG] Assembly element types: inner={etype['inner']}, boundary={etype['boundary']}, outer={etype['outer']}")
    
    # Apply WEB transform
    start_time = time.time()
    print(f"    [DEBUG] Before WEB transform: mesh.TRANSFORM={mesh.TRANSFORM}")
    K, F, etype, bsp_class, ext_basis, E_tilde = FEM_WEB.transformStandardSystemToWEB(
        K, F, model, p, q, knotvector_u, knotvector_w, xDivision, yDivision,
        web_use_weight_normalization=True,
        web_ref_weight_eps=1e-6,
        extension_method="collocation"
    )
    transform_time = time.time() - start_time
    print(f"    [DEBUG] After WEB transform: n_inner={bsp_class['n_inner']}, n_outer={bsp_class['n_outer']}")
    
    # Solve
    start_time = time.time()
    result = FEM_WEB.solveWEB(K, F)
    solve_time = time.time() - start_time
    print(f"    [DEBUG] Solution computed: len(result)={len(result)}, min={np.min(result):.6f}, max={np.max(result):.6f}")
    
    # Test reconstruction at a sample point to verify transformation is used
    test_x, test_y = 0.5, 0.5
    reconstructed_value = FEM_WEB.reconstructSolution(
        test_x, test_y, result, model, p, q, knotvector_u, knotvector_w,
        bsp_class, ext_basis
    )
    print(f"    [DEBUG] Reconstructed value at ({test_x}, {test_y}): {reconstructed_value:.6f}")
    
    # Evaluate errors
    print(f"    [DEBUG] Evaluating errors with mesh.TRANSFORM={mesh.TRANSFORM}")
    metrics_full = evaluation_WEB.evaluateAccuracyWEB(
        model, result, p, q, knotvector_u, knotvector_w,
        bspline_classification=bsp_class, extended_basis=ext_basis,
        N=n_samples, seed=42
    )
    
    metrics_L2H1 = evaluation_WEB.computeL2andH1Errors(
        model, result, p, q, knotvector_u, knotvector_w,
        bspline_classification=bsp_class, extended_basis=ext_basis,
        N=n_samples, seed=42
    )
    
    # Combine metrics and record actual transform settings used
    result_dict = {
        'assembly_time': assembly_time,
        'transform_time': transform_time,
        'solve_time': solve_time,
        'total_time': assembly_time + transform_time + solve_time,
        'n_inner': bsp_class['n_inner'],
        'n_outer': bsp_class['n_outer'],
        'n_valid_points': metrics_full['n_valid_points'],
        'MSE': metrics_full['MSE'],
        'MAE': metrics_full['MAE'],
        'L_inf': metrics_full['L_inf'],
        'relative_error': metrics_full['relative_error'],
        'L2_error': metrics_L2H1['L2_error'],
        'H1_error': metrics_L2H1['H1_error'],
        'H1_seminorm': metrics_L2H1['H1_seminorm'],
        # Record what transform was actually used during simulation
        'actual_transform': mesh.TRANSFORM,
        'actual_delta_hollig': mesh.DELTA_HOLLIG if hasattr(mesh, 'DELTA_HOLLIG') else None,
        'actual_gamma_hollig': mesh.GAMMA_HOLLIG if hasattr(mesh, 'GAMMA_HOLLIG') else None,
        # Add K matrix characteristics for debugging
        'K_norm': float(K_norm),
        'K_trace': float(K_trace),
        'solution_min': float(np.min(result)),
        'solution_max': float(np.max(result)),
        'solution_mean': float(np.mean(result)),
    }
    
    print(f"    [DEBUG] Results: MAE={result_dict['MAE']:.6e}, L_inf={result_dict['L_inf']:.6e}")
    print(f"    [DEBUG] Solution stats: min={result_dict['solution_min']:.6f}, max={result_dict['solution_max']:.6f}, mean={result_dict['solution_mean']:.6f}")
    
    return result_dict


def plotHolligStudyResults(
    json_filename,
    error_metric='MAE',
    gamma_values=None,
    figsize=(14, 8)
):
    """
    Visualize results from Höllig transformation parameter study.
    
    Plots error vs delta for different gamma values.
    
    Parameters:
    -----------
    json_filename : str
        Path to JSON file with study results
    error_metric : str
        Error metric to plot: 'MAE', 'L_inf', 'L2_error', 'H1_error', etc.
    gamma_values : list, optional
        Specific gamma values to plot. If None, plots all.
    figsize : tuple
        Figure size (width, height)
    """
    # Load results
    with open(json_filename, 'r') as f:
        results = json.load(f)
    
    runs = results['runs']
    metadata = results['metadata']
    
    # Get baseline (no transformation)
    baseline_runs = [r for r in runs if r['transform'] is None]
    baseline_error = baseline_runs[0][error_metric] if baseline_runs else None
    
    # Filter to Höllig transformation runs
    hollig_runs = [r for r in runs if r['transform'] == 'Hollig']
    
    # Extract unique gamma values
    all_gammas = sorted(set(r['gamma'] for r in hollig_runs))
    if gamma_values is not None:
        all_gammas = [g for g in all_gammas if g in gamma_values]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Color map for different gamma values
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_gammas)))
    
    # Plot 1: Absolute error vs delta
    for idx, gamma in enumerate(all_gammas):
        gamma_runs = [r for r in hollig_runs if r['gamma'] == gamma]
        gamma_runs = sorted(gamma_runs, key=lambda x: x['delta'])
        
        deltas = [r['delta'] for r in gamma_runs]
        errors = [r[error_metric] for r in gamma_runs]
        
        ax1.plot(deltas, errors, marker='o', label=f'γ = {gamma}',
                color=colors[idx], linewidth=2, markersize=8)
    
    if baseline_error is not None:
        ax1.axhline(baseline_error, color='red', linestyle='--', linewidth=2,
                   label='Baseline (no transform)', zorder=1)
    
    ax1.set_xlabel('δ (delta)', fontsize=12)
    ax1.set_ylabel(f'{error_metric}', fontsize=12)
    ax1.set_title(f'{error_metric} vs Delta for Different Gamma Values', fontsize=13, weight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add vertical lines for element width multiples
    element_width = metadata['element_width']
    for mult in [1, 2, 3, 4, 5]:
        ax1.axvline(mult * element_width, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax1.text(mult * element_width, ax1.get_ylim()[1], f'{mult}×elem',
                rotation=90, va='top', ha='right', fontsize=8, alpha=0.6)
    
    # Plot 2: Relative improvement vs delta
    if baseline_error is not None:
        for idx, gamma in enumerate(all_gammas):
            gamma_runs = [r for r in hollig_runs if r['gamma'] == gamma]
            gamma_runs = sorted(gamma_runs, key=lambda x: x['delta'])
            
            deltas = [r['delta'] for r in gamma_runs]
            improvements = [(baseline_error - r[error_metric]) / baseline_error * 100
                          for r in gamma_runs]
            
            ax2.plot(deltas, improvements, marker='o', label=f'γ = {gamma}',
                    color=colors[idx], linewidth=2, markersize=8)
        
        ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('δ (delta)', fontsize=12)
        ax2.set_ylabel('Improvement vs Baseline (%)', fontsize=12)
        ax2.set_title('Relative Improvement vs Delta', fontsize=13, weight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for element width multiples
        for mult in [1, 2, 3, 4, 5]:
            ax2.axvline(mult * element_width, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    fig.suptitle(
        f'Höllig Transformation Study: {metadata["geometry"]}, '
        f'divisions={metadata["divisions"]}, order={metadata["order"]}',
        fontsize=14, weight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Error metric: {error_metric}")
    if baseline_error is not None:
        print(f"Baseline error (no transform): {baseline_error:.6e}")
    print()
    
    for gamma in all_gammas:
        gamma_runs = [r for r in hollig_runs if r['gamma'] == gamma]
        best_run = min(gamma_runs, key=lambda x: x[error_metric])
        
        print(f"γ = {gamma}:")
        print(f"  Best error: {best_run[error_metric]:.6e} at δ = {best_run['delta']:.6f}")
        if baseline_error is not None:
            improvement = (baseline_error - best_run[error_metric]) / baseline_error * 100
            print(f"  Improvement: {improvement:+.2f}%")
        print()
    
    print("="*80)


def PlotSubdivide(x1,x2,y1,y2,ax,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    halfy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    gpx1 = halfx-w/2 *(1 / math.sqrt(3))
    gpy1 = halfy-h/2 *(1 / math.sqrt(3))

    gpx2 = halfx+w/2 *(1 / math.sqrt(3))
    gpy2 = halfy+h/2 *(1 / math.sqrt(3))
    
    if isBoundary(x1,x2,y1,y2) and level<MAXLEVEL:
        plotRectangle(x1, halfx,y1,halfy,level,ax)
        plotRectangle(x1, halfx,halfy,y2,level,ax)
        plotRectangle(halfx, x2,y1,halfy,level,ax)
        plotRectangle(halfx, x2,halfy,y2,level,ax)
    else:
        if not isOutside(x1,x2,y1,y2):
            ax.plot(gpx1,gpy1,'bx')
            ax.plot(gpx1,gpy2,'bx')
            ax.plot(gpx2,gpy1,'bx')
            ax.plot(gpx2,gpy2,'bx')
        return 0
    if level == MAXLEVEL:
        if not isOutside(x1,x2,y1,y2):
            ax.plot(gpx1,gpy1,'bx')
            ax.plot(gpx1,gpy2,'bx')
            ax.plot(gpx2,gpy1,'bx')
            ax.plot(gpx2,gpy2,'bx')
    else:
        PlotSubdivide(x1,halfx,y1,halfy,ax,level+1,MAXLEVEL)
        PlotSubdivide(x1,halfx,halfy,y2,ax,level+1,MAXLEVEL)
        PlotSubdivide(halfx,x2,y1,halfy,ax,level+1,MAXLEVEL)
        PlotSubdivide(halfx,x2,halfy,y2,ax,level+1,MAXLEVEL)
    

def plotRectangle(x1,x2,y1,y2,level,ax):
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
def isBoundary(x1,x2,y1,y2):
    distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]

    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point>1:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        return False
    elif outerElement:
        return False
    else:
        return True
def isOutside(x1,x2,y1,y2):
    distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]

    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point>1:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        return False
    elif outerElement:
        return True
    else:
        return False
    
if __name__ == "__main__":
    results = runHolligTransformationStudy(
        divisions=50,
        order=2,
        delta_sim=0.005,
        n_samples=2000,
        save_filename='hollig_study_20260130_test_L_shape.json'
    )
    # Plot MAE
    plotHolligStudyResults('hollig_study_20260130_test_L_shape.json', error_metric='MAE')

    # Plot L_inf for specific gamma values
    plotHolligStudyResults('hollig_study_20260130_test_L_shape.json', 
                        error_metric='L_inf', 
                        gamma_values=[2, 3, 4])

    # Plot L2 or H1 errors
    plotHolligStudyResults('hollig_study_20260130_test_L_shape.json', error_metric='L2_error')
