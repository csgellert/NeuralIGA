import numpy as np
import matplotlib.pyplot as plt
import FEM
from scipy.interpolate import BSpline
from math import exp, sqrt,sin
print("\n\nSolving one dimensional Laplace equation with IGA.\n","-"*50,"\n")
print("10 * d2u/dx2 = sin(x)\nu(0)=1\nu(0.5)=0")
#Endpoints of the domain
xAe = 0
xBe = 10
xA = xAe-0.1
xB = xBe +0.1

uAe = 1
uBe = 0
l = xBe-xAe # length of the domain

#Number of points
N =30
p = 1 

# defining the domain 
knots = np.linspace(xA, xB, N)
knots = np.insert(knots,0,[xA for i in range(p)])
knots = np.append(knots,[xB for i in range(p)])

x = np.linspace(xAe,xBe,1000)

from utility import plotBsplineBasis
plotBsplineBasis(x,knots,p,True)
K = np.zeros((N+p-1,N+p-1))
F = np.zeros(N+p-1)
for i in range(N-1):
    #print(i)
    Ke,Fe = FEM.element(i,knots,p)
    K,F = FEM.assembly(K,F,Ke,Fe,i)
print(K[1:2,1:3])
print("")
print(F)
u = FEM.solve(F,K)
analitical = [(-sin(xv) + ((1/xBe)*(sin(xBe)-10))*xv +10)/10 for xv in x]
print("Mean Absolute Error:",np.mean(np.abs(FEM.evaluate(u,knots,p,x)[:-1]-analitical[:-1])))
FEM.show(u,knots,p,x, analitical,plotbasis=True)






