import numpy as np
import FEM
import mesh
from tqdm import tqdm
import time
import NeuralImplicit 
import cProfile
from pstats import Stats
import Geomertry

model = NeuralImplicit.load_models("siren_model")
r=1
DIVISIONS = 5
ORDER = 2
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
#mesh.plotMesh(xDivision,yDivision,delta=0.005)
#mesh.plotAlayticHeatmap(FEM.solution_function)
Geomertry.init_spl(x,p,None,knotvector_u)

K = np.zeros(((xDivision+p+1)*(yDivision+q+1),(xDivision+p+1)*(yDivision+q+1)))
F = np.zeros((xDivision+p+1)*(yDivision+q+1))
print("Initialisation finished")
with cProfile.Profile() as pr:
    for elemx in tqdm(range(p,p+xDivision+1)):
        for elemy in range(q,q+xDivision+1):
            Ke,Fe = FEM.elementChoose(model,False,r,p,q,knotvector_u,knotvector_w,None,elemx,elemy,NControl_u,NControl_w,weigths,None)
            K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
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
#FEM.plotErrorHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=False,N=40)
#FEM.plotResultHeatmap(model,result,knotvector_u,knotvector_w,p,q,larger_domain=FEM.LARGER_DOMAIN,N=50)


FEM.visualizeResultsBspline(model,result,p,q,knotvector_u,knotvector_w,None,larger_domain=FEM.LARGER_DOMAIN)



