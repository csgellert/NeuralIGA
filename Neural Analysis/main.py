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
#torch.set_default_dtype(torch.float64)

#model = load_test_model("SIREN_circle", "SIREN", params={"architecture": [2, 256, 256, 256, 1], "w_0": 15.0, "w_hidden": 30.0})
model = Geomertry.AnaliticalDistanceCircle()
DIVISIONS = 100
ORDER = 1
DELTA = 0.005


#defining geometry:
default = mesh.getDefaultValues(div=DIVISIONS,order=ORDER,delta=DELTA,larger_domain=FEM.LARGER_DOMAIN)
x0, y0,x1,y1,xDivision,yDivision,p,q = default
knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
assert p==q and xDivision == yDivision
x = np.linspace(x0,x1,10)
y = np.linspace(y0,y1,10)
NControl_u = len(knotvector_u)-p-1
NControl_w = len(knotvector_w)-q-1
mesh.plotMesh(xDivision,yDivision,delta=0.005)
#mesh.plotAlayticHeatmap(FEM.solution_function)
mesh.plotDisctancefunction(model)
Geomertry.init_spl(x,p,None,knotvector_u)

K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
print("Initialisation finished")
start = time.time()
with cProfile.Profile() as pr:
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
result = FEM.solveWeak(K,F)
print(f"Calculation time: {time.time()-start} ms")
print(f"p={p}\tdiv = {xDivision}")
evaluation.plotErrorHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=False,N=40)
#evaluation.plotResultHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=FEM.LARGER_DOMAIN,N=50)


evaluation.visualizeResultsBspline(model,result,p,q,knotvector_u,knotvector_w)

metrics = evaluation.evaluateAccuracy(model, result, p, q, knotvector_u, knotvector_w, N=10000, seed=42)
evaluation.printErrorMetrics(metrics)  # Pretty print all metrics
