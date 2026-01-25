import numpy as np
import torch
import evaluation

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)

import FEM
import mesh
import time
import cProfile
from pstats import Stats
import Geomertry
from network_defs import load_test_model
import FEM_WEB
import evaluation_WEB
#torch.set_default_dtype(torch.float64)

#model = load_test_model("SIREN_circle", "SIREN", params={"architecture": [2, 256, 256, 256, 1], "w_0": 15.0, "w_hidden": 30.0})
model = Geomertry.AnaliticalDistanceCircle()
DIVISIONS = 100
ORDER = 3
DELTA = 0.005
USE_WEB =False
USE_WEB_TRANSFORM = False
USE_WEB_DIAG_EXTRACT = True
assert sum([bool(USE_WEB), bool(USE_WEB_TRANSFORM), bool(USE_WEB_DIAG_EXTRACT)]) <= 1, \
    "USE_WEB / USE_WEB_TRANSFORM / USE_WEB_DIAG_EXTRACT are mutually exclusive"


#defining geometry:
default = mesh.getDefaultValues(div=DIVISIONS,order=ORDER,delta=DELTA)
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
assert p==q and xDivision == yDivision
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1
#mesh.plotMesh(xDivision,yDivision,delta=0.005)
#mesh.plotAlayticHeatmap(FEM.solution_function)
#mesh.plotDisctancefunction(model)
Geomertry.init_spl(x,p,None,knotvector_u)

K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
print("Initialisation finished")
start = time.time()
with cProfile.Profile() as pr:
    if USE_WEB:
        print("Assembling using WEB/Höllig method")
        K, F, etype, bsp_class, ext_basis= FEM_WEB.processAllElementsWEB(model,p,q,knotvector_u,knotvector_w,
                                              xDivision,yDivision,extension_strict=True,web_use_weight_normalization=False)
    elif USE_WEB_TRANSFORM:
        print("Assembling using standard weighted B-splines")
        K, F, etype = FEM.processAllElements(model, p, q, knotvector_u, knotvector_w, 
                                      xDivision, yDivision, K, F)
        print("Applying WEB coupling matrix transform (Eq. 8.9)")
        K, F, etype, bsp_class, ext_basis, E_tilde = FEM_WEB.transformStandardSystemToWEB(
            K,
            F,
            model,
            p,
            q,
            knotvector_u,
            knotvector_w,
            xDivision,
            yDivision,
            extension_strict=True,
            web_use_weight_normalization=False,
            web_ref_weight_eps=1e-6,
        )
    elif USE_WEB_DIAG_EXTRACT:
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
            diag_threshold=1e-10,
            diag_nonzero_eps=0.0,
            extension_strict=True,
        )
    else:
        print("Assembling using standard method")
        K, F, etype = FEM.processAllElements(model, p, q, knotvector_u, knotvector_w, 
                                      xDivision, yDivision, K, F)
end = time.time()
print(f"Assembly time: {end - start} ms")
with open('profiling_stats.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('.prof_stats')
    stats.print_stats()

start = time.time()
print("Solving equation")
if USE_WEB:
    print("Solving using WEB/Höllig method")
    result = FEM_WEB.solveWEB(K,F)
    print(f"Calculation time: {time.time()-start} ms")
    print(f"p={p}\tdiv = {xDivision}")
    #evaluation_WEB.plotErrorHeatmapWEB(model,result,knotvector_u,knotvector_w,p,q,bspline_classification=bsp_class,extended_basis=ext_basis)
    #evaluation_WEB.plotSolutionHeatmapWEB(model,result,knotvector_u,knotvector_w,p,q,bspline_classification=bsp_class,extended_basis=ext_basis,N=100)
    metrics = evaluation_WEB.evaluateAccuracyWEB(model, result, p, q, knotvector_u, knotvector_w, bspline_classification=bsp_class, extended_basis=ext_basis, N=10000, seed=42)
    evaluation_WEB.printErrorMetricsWEB(metrics)  # Pretty print all metrics
elif USE_WEB_TRANSFORM:
    print("Solving using WEB/Höllig method after matrix transform")
    result = FEM_WEB.solveWEB(K,F)
    print(f"Calculation time: {time.time()-start} ms")
    print(f"p={p}\tdiv = {xDivision}")
    evaluation_WEB.plotErrorHeatmapWEB(model,result,knotvector_u,knotvector_w,p,q,bspline_classification=bsp_class,extended_basis=ext_basis)
    evaluation_WEB.plotSolutionHeatmapWEB(model,result,knotvector_u,knotvector_w,p,q,bspline_classification=bsp_class,extended_basis=ext_basis,N=100)
    metrics = evaluation_WEB.evaluateAccuracyWEB(model, result, p, q, knotvector_u, knotvector_w, bspline_classification=bsp_class, extended_basis=ext_basis, N=10000, seed=42)
    evaluation_WEB.printErrorMetricsWEB(metrics)  # Pretty print all metrics
elif USE_WEB_DIAG_EXTRACT:
    print("Solving using selective diagonal-based extraction")
    result = FEM_WEB.solveWEB(K,F)
    print(f"Calculation time: {time.time()-start} ms")
    print(f"p={p}\tdiv = {xDivision}")
    print("Transforming solution back to standard B-spline basis")
    results_v2 = E_tilde.T @ result
    metrics_v2 = evaluation.evaluateAccuracy(model, results_v2, p, q, knotvector_u, knotvector_w, N=10000, seed=42)
    evaluation.printErrorMetrics(metrics_v2)  # Pretty print all metrics
else:
    print("Solving using standard method")
    result = FEM.solveWeak(K,F)
    print(f"Calculation time: {time.time()-start} ms")
    print(f"p={p}\tdiv = {xDivision}")
    evaluation.plotErrorHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=False,N=40)
    #evaluation.plotResultHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=FEM.LARGER_DOMAIN,N=50)
    evaluation.visualizeResultsBspline(model,result,p,q,knotvector_u,knotvector_w)

    metrics = evaluation.evaluateAccuracy(model, result, p, q, knotvector_u, knotvector_w, N=10000, seed=42)
    evaluation.printErrorMetrics(metrics)  # Pretty print all metrics
