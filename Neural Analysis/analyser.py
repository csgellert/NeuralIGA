import numpy as np
import torch
import evaluation
import json
from datetime import datetime

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)

import FEM
import mesh
import Geomertry
from network_defs import load_test_model


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


#mdl_name = "SIREN_circle_1_2" #! check network params!!
#model = load_test_model(mdl_name, "SIREN", params={"architecture": [2, 256, 256, 256, 1], "w_0": 15.0, "w_hidden": 15.0})
mdl_name = "AnalyticalDistanceCircleTanh"
model = Geomertry.AnaliticalDistanceCircle()
FEM.FUNCTION_CASE = 3
FEM.MAX_SUBDIVISION = 4
mesh.TRANSFORM = 'tanh'

test_values = [10, 20,50,80,100]
orders = [1,2,3]

esize = [1/(nd+1) for nd in test_values]

all_eval_stats = {}  # Store all evaluation stats

for order in orders:                                                                            
    accuracy = []
    etypes = []
    eval_stats = []
    for division in test_values:
        print(f"Running simulation for order={order}, division={division}")
        default = mesh.getDefaultValues(div=division,order=order,delta=0.005,larger_domain=FEM.LARGER_DOMAIN)
        x0, y0,x1,y1,xDivision,yDivision,p,q = default
        knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
        assert p==q and xDivision == yDivision

        x = np.linspace(x0,x1,10)
        y = np.linspace(y0,y1,10)
        NControl_u = len(knotvector_u)-p-1
        NControl_w = len(knotvector_w)-q-1
        Geomertry.init_spl(x,p,None,knotvector_u)
        
        K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
        F = np.zeros((xDivision+p+1)*(yDivision+q+1))
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
    max_subdivision=FEM.MAX_SUBDIVISION
)
