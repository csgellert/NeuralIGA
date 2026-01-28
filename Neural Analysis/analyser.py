import numpy as np
import torch
import evaluation
import json
from datetime import datetime
from scipy import sparse

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)

import FEM
import mesh
import Geomertry
from network_defs import load_test_model
import FEM_WEB
import evaluation_WEB


def save_simulation_results(filename, mdl_name, test_values, orders, all_eval_stats, 
                            function_case=None, max_subdivision=None):
    """
    Save simulation results to a JSON file.
    
    Parameters:
    -----------
    filename : str
        Path to the output file
    mdl_name : str
        Name of the model used
    test_values : list
        List of division values tested
    orders : list
        List of B-spline orders tested
    all_eval_stats : dict
        Dictionary with keys as (order, division) tuples and values as eval_stats dicts
    function_case : int, optional
        The FEM.FUNCTION_CASE parameter
    max_subdivision : int, optional
        The FEM.MAX_SUBDIVISION parameter
    """
    # Convert tuple keys to string keys for JSON compatibility
    results_serializable = {}
    for (order, division), stats in all_eval_stats.items():
        key = f"order_{order}_div_{division}"
        # Convert numpy types to Python native types
        results_serializable[key] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                                      for k, v in stats.items()}
    
    data = {
        "model_name": mdl_name,
        "test_values": test_values,
        "orders": orders,
        "function_case": function_case,
        "max_subdivision": max_subdivision,
        "transform": mesh.TRANSFORM,
        "timestamp": datetime.now().isoformat(),
        "eval_stats": results_serializable
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filename}")


#mdl_name = "SIREN_circle_1_4" #! check network params!!
#model = load_test_model(mdl_name, type="SIREN", params={"architecture": [2, 256, 256, 256, 1], "w_0": 8.0, "w_hidden": 8.0})
mdl_name = "AnalyticalDistance_Circle_WEB"
model = Geomertry.AnaliticalDistanceCircle()
#model = Geomertry.AnaliticalDistanceLshape()
USE_WEB = True
WEB_ADAPTIVE = False
DIAG_TRSH = 1e-9
assert not (USE_WEB and WEB_ADAPTIVE), "USE_WEB and WEB_ADAPTIVE are mutually exclusive"


test_values = [10, 20, 50, 80,95,100,105,120]
orders = [1, 2, 3]

esize = [1/(nd+1) for nd in test_values]

all_eval_stats = {}  # Store all evaluation stats

for order in orders:                                                                            
    accuracy = []
    etypes = []
    eval_stats = []
    for division in test_values:
        print(f"Running simulation for order={order}, division={division}")
        default = mesh.getDefaultValues(div=division,order=order,delta=0.005)
        x0, y0,x1,y1,xDivision,yDivision,p,q = default
        knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
        assert p==q and xDivision == yDivision

        x = np.linspace(x0,x1,10)
        y = np.linspace(y0,y1,10)
        NControl_u = len(knotvector_u)-p-1
        NControl_w = len(knotvector_w)-q-1
        Geomertry.init_spl(x,p,None,knotvector_u)
        
        # Use sparse matrix format (lil_matrix is efficient for construction)
        matrix_size = (xDivision+p+1)*(yDivision+q+1)
        K = sparse.lil_matrix((matrix_size, matrix_size), dtype=np.float64)
        F = np.zeros(matrix_size)
        if USE_WEB:
            # Assemble and solve with WEB-splines
            K, F, etype, bsp_class, ext_basis = FEM_WEB.processAllElementsWEB(
                model, p, q, knotvector_u, knotvector_w, xDivision, yDivision
            )
            u = FEM_WEB.solveWEB(K, F)


            eval_stats = evaluation_WEB.evaluateAccuracyWEB(
                model, u, p, q, knotvector_u, knotvector_w, bsp_class, ext_basis, N=10000, seed=42)
            evaluation_WEB.printErrorMetricsWEB(eval_stats)
        elif WEB_ADAPTIVE:
            print("Assembling using standard weighted B-splines")
            K, F, etype = FEM.processAllElements(model, p, q, knotvector_u, knotvector_w,
                                        xDivision, yDivision, K, F)
            print("Applying selective diagonal-based extraction (partial WEB transform)")
            K, F, etype, diag_meta, E_tilde = FEM_WEB.transformStandardSystemToWEBSelectiveDiagonalExtraction(
                K,
                F,
                model,
                p,
                q,
                knotvector_u,
                knotvector_w,
                xDivision,
                yDivision,
                diag_threshold=DIAG_TRSH,
                diag_nonzero_eps=0.0,
                extension_strict=True,
            )
            print("Solving using selective diagonal-based extraction")
            result = FEM_WEB.solveWEB(K,F)
            print("Transforming solution back to standard B-spline basis")
            results_v2 = E_tilde.T @ result
            eval_stats = evaluation.evaluateAccuracy(model, results_v2, p, q, knotvector_u, knotvector_w, N=10000, seed=42)
            evaluation.printErrorMetrics(eval_stats)  # Pretty print all metrics
        else:
            K, F, etype = FEM.processAllElements(model, p, q, knotvector_u, knotvector_w, xDivision, yDivision, K, F)

            result = FEM.solveWeak(K,F)

            eval_stats = evaluation.evaluateAccuracy(model, result, p, q, knotvector_u, knotvector_w, N=10000, seed=42)
            accuracy.append(eval_stats["MAE"])
            evaluation.printErrorMetrics(eval_stats)
        
        # Store results for this configuration
        all_eval_stats[(order, division)] = eval_stats

# Save all results to file
save_simulation_results(
    filename=f"simulation_results_{mdl_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mdl_name=mdl_name,
    test_values=test_values,
    orders=orders,
    all_eval_stats=all_eval_stats,
    function_case=FEM.FUNCTION_CASE,
    max_subdivision=FEM.MAX_SUBDIVISION,
)
