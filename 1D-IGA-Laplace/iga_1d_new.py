import numpy as np
import matplotlib.pyplot as plt
import FEM
from utility import B, dBdXi
print("\n\nSolving one dimensional Laplace equation with IGA.\n","-"*50,"\n")
print("Laplace equation:\t\t-d^2 f(x) / dx^2 = 6 - 6x + 12x^2 -20x^3\n\n")
print("RB1: f(0) = 1\nf'(2)=50")
#Endpoints of the domain
xA = 0
xB = 2
l = xB-xA # length of the domain

#Number of points
N = 5
p = 1 #quadratic B-splines

# defining the domain 
knots = np.linspace(xA, xB, N)
knots = np.insert(knots,0,[xA for i in range(p)])
knots = np.append(knots,[xB for i in range(p)])

x = np.linspace(xA,xB,10)
grad = len(knots)-p-1
ctrl = np.linspace(knots[0],knots[-1],grad)
x_orig = np.zeros(x.shape)
for i in range(grad):
    x_orig += ctrl[i]*np.array([B(xx,p,i,knots) for xx in x])

from utility import plotBsplineBasis
#plotBsplineBasis(x,knots,p)

K,F = FEM.element(x,knots,p)
K_o,F_o = FEM.element_old(x,knots,p)
u = FEM.solve(F,K)
u_o = FEM.solve(F_o,K_o)
analitical = [xv**5-xv**4+xv**3-3*xv*xv+2*xv+1 for xv in x]
analitical_new = [xv**5-xv**4+xv**3-3*xv*xv+2*xv+1 for xv in x_orig]
plt.plot(x_orig,analitical_new)
plt.show()
print("Mean Absolute Error:",np.mean(np.abs(FEM.evaluate(u,knots,p,x)[:-1]-analitical[:-1])))
FEM.show(u,knots,p,x, analitical_new,plotbasis=True)
FEM.show_old(u_o,knots,p,x, analitical,plotbasis=True)






