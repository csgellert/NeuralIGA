import numpy as np
import FEM
import mesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import Geomertry
import NeuralImplicit

model = NeuralImplicit.load_models("analitical_model")
model = model

r=1

test_values = [110]
esize = [1/(nd+1) for nd in test_values]
orders = [2]
fig,ax = plt.subplots()
for order in orders:                                                                            
    accuracy = []
    etypes = []
    for division in test_values:
        etype = {"outer":0,"inner":0,"boundary":0}
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
        for elemx in tqdm(range(p,p+xDivision+1)):
            for elemy in range(q,q+xDivision+1):
                Ke,Fe,etype = FEM.elementChoose(model,p,q,knotvector_u,knotvector_w,elemx,elemy,etype)
                K,F = FEM.assembly(K,F,Ke,Fe,elemx,elemy,p,q,xDivision,yDivision)
        #print(dirichlet)
        result = FEM.solveWeak(K,F)
        accuracy.append(FEM.calculateErrorBspline(model,result,p,q,knotvector_u, knotvector_w,larger_domain=FEM.LARGER_DOMAIN))
        etypes.append(etype)
    #ax.semilogy(test_values,accuracy)
    ax.loglog(esize,accuracy)
ax.set_title("Convergence of MSE based on number of elements")
#ax.set_xlabel("Number of divisions")
ax.set_xlabel("log(Element size)")
ax.set_ylabel("log(Mean Square Error)")
ax.legend(["p=1","p=2","p=3"])

plt.show()
inner = np.array([case["inner"] for case in etypes])
outer = np.array([case["outer"] for case in etypes])
boundary = np.array([case["boundary"] for case in etypes])
print(etypes)
fig,ax = plt.subplots()
ax.loglog(esize,(inner+boundary)/(inner+boundary+outer))
plt.show()