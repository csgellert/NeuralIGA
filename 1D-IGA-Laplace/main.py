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

#Number of elements
N = 5
p = 2 #quadratic B-splines

# defining the domain 
knots = np.linspace(xA, xB, N)
knots = np.insert(knots,0,[xA,xA])
knots = np.append(knots,[xB,xB])
#knots = [0, 0, xB, xB]
#print(knots)

c = np.ones(N)
#print(BSpline(knots,c,p))
x = np.linspace(xA,xB,1000)

K,F = FEM.element(x,knots,p)
u = FEM.solve(F,K)
analitical = [xv**5-xv**4+xv**3-3*xv*xv+2*xv+1 for xv in x]
FEM.show(u,knots,p,x, analitical)







