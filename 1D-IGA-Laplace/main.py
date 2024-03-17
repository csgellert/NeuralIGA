import numpy as np
import matplotlib.pyplot as plt
import FEM
from scipy.interpolate import BSpline
print("\n\nSolving one dimensional Laplace equation with IGA.\n","-"*50,"\n")
print("Laplace equation:\t\t-d^2 f(x) / dx^2 = 6 - 6x + 12x^2 -20x^3\n\n")
print("RB1: f(0) = 1\nf'(2)=50")
#Endpoints of the domain
xA = 0
xB = 2
l = xB-xA # length of the domain

#Number of points
N = 4
p = 4 #quadratic B-splines

# defining the domain 
knots = np.linspace(xA, xB, N)
knots = np.insert(knots,0,[xA for i in range(p)])
knots = np.append(knots,[xB for i in range(p)])

x = np.linspace(xA,xB,1000)

from utility import plotBsplineBasis
plotBsplineBasis(x,knots,p)

K,F = FEM.element(x,knots,p)
u = FEM.solve(F,K)
analitical = [xv**5-xv**4+xv**3-3*xv*xv+2*xv+1 for xv in x]
print("Mean Absolute Error:",np.mean(np.abs(FEM.evaluate(u,knots,p,x)[:-1]-analitical[:-1])))
FEM.show(u,knots,p,x, analitical,plotbasis=True)






