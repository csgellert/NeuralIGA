import utility
import FEM
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tqdm
LR = 0.0001
ITER = 100


def Err(u,x,xi,p):
    return 10*FEM.ValuePointD2(u,xi,p,x) - math.sin(x)
def dErrdUi(i,x,xi,p):
    return 10*(utility.distance2ndDerivative(x)*utility.B(x,p,i,xi) + 2*utility.distanceDerivative(x)*utility.dBdXi(x,p,i,xi)+ utility.disctance(x)*utility.d2BdXi2(x,p,i,xi))
def dErr2(u,x,xi,p,i):
    E = Err(u,x,xi,p)
    Ed = dErrdUi(i,x,xi,p)
    return E*Ed, E
if __name__ == "__main__":

    Erro = []

    xAe = 0
    xBe = 10
    xA = xAe - 0.1
    xB = xBe + 0.1
    uAe = 0#1
    uBe = 0
    l = xBe-xAe # length of the domain
    #Number of points
    N = 6
    p = 3 
    # defining the domain 
    knots = np.linspace(xA, xB, N)
    knots = np.insert(knots,0,[xA for i in range(p)])
    knots = np.append(knots,[xB for i in range(p)])
    u = np.zeros(N+p-1)
    evl_points = []
    for iter in tqdm.tqdm(range(ITER)):
        x = random.random()*l+xA
        evl_points.append(x)
        for i in range(0,len(u)):
            err_grad, error = dErr2(u,x,knots,p,i)
            u[i] -= LR*err_grad
        #Erro.append(error**2)
    
    ITER = 10000
    LR = 0.0005
    for iter in tqdm.tqdm(range(ITER)):
        x = random.random()*l+xA
        evl_points.append(x)
        for i in range(0,len(u)):
            err_grad, error = dErr2(u,x,knots,p,i)
            u[i] -= LR*err_grad
        Erro.append(error**2)
    x = np.linspace(xAe,xBe,1000)
    analitical = [(-math.sin(xv) + ((1/xBe)*(math.sin(xBe)-0))*xv +0 )/10 -FEM.u_star(xv,xAe,xBe,0,0) for xv in x]
    plt.plot(evl_points)
    plt.show()
    plt.plot(Erro)
    plt.show()
    FEM.show(u,knots,p,x,analitical,plotbasis=True)
