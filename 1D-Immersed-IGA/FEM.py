from utility import *
import numpy as np
import matplotlib.pyplot as plt
import math
def element(i, knotvector, p):
    DO_SUBDIV = True
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    Ke = np.zeros((p+1,p+1))
    Fe = np.zeros(p+1)
    x1 = knotvector[i+p]
    x2 = knotvector[i+1+p]
    J = 1#(x2-x1)/2
    Ji = 1/J
    if DO_SUBDIV and (x1 <= 0 or x2>=10):
        Ket, Fet = Subdivide(x1,x2,i,knotvector,p,0,MAXLEVEL=8)
        Ke+= Ket
        Fe+= Fet 
        return Ke, Fe
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        xi = (x2-x1)/2 * gpx + (x2+x1)/2
        d = disctance(xi)
        if xi <0 or xi>10: continue
        dd = distanceDerivative(xi)
        for xbasisi in range(i,i+1+p):
            for xbasisj in range(i,i+p+1):
                dgammai = d*dBdXi(xi,p,xbasisi,knotvector) + dd*B(xi,p,xbasisi,knotvector)
                dgammaj = d*dBdXi(xi,p,xbasisj,knotvector) + dd*B(xi,p,xbasisj,knotvector)
                Ke[xbasisi-(i),xbasisj-(i)] += (-10)*dgammai*dgammaj
        for xbasisi in range(i,i+p+1):
            f = math.sin(xi)
            Fe[xbasisi-(i)] += f*d*B(xi,p,xbasisi,knotvector)
    return Ke, Fe
def Subdivide(x1,x2,i,knotvector,p,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    K = np.zeros(((p+1),(p+1)))
    F = np.zeros((p+1))
    if level == MAXLEVEL:
        #first
        Ks,Fs = GaussQuadrature(x1, halfx,i,p,knotvector)
        K+=Ks/2
        F+=Fs/2
        #second
        Ks,Fs = GaussQuadrature(halfx, x2,i,p,knotvector)
        K+=Ks/2
        F+=Fs/2
        return K,F
    else:
        Kret,Fret=Subdivide(x1,halfx,i,knotvector,p,level+1,MAXLEVEL)
        K+= Kret/2
        F+=Fret/2
        Kret,Fret=Subdivide(halfx,x2,i,knotvector,p,level+1,MAXLEVEL)
        K+= Kret/2
        F+=Fret/2
        return K,F
def GaussQuadrature(x1,x2,i,p,knotvector):
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    Ke = np.zeros(((p+1),(p+1)))
    Fe = np.zeros((p+1))
    J = 1#(x2-x1)/2
    for idxx,gpx in enumerate(g): #iterate throug Gauss points functions in x direction
        xi = (x2-x1)/2 * gpx + (x2+x1)/2
        d = disctance(xi)
        if xi <0 or xi>10: continue
        dd = distanceDerivative(xi)
        for xbasisi in range(i,i+1+p):
            for xbasisj in range(i,i+p+1):
                dgammai = d*dBdXi(xi,p,xbasisi,knotvector) + dd*B(xi,p,xbasisi,knotvector)
                dgammaj = d*dBdXi(xi,p,xbasisj,knotvector) + dd*B(xi,p,xbasisj,knotvector)
                Ke[xbasisi-(i),xbasisj-(i)] += (-10)*dgammai*dgammaj*J
        for xbasisi in range(i,i+p+1):
            f = math.sin(xi)
            Fe[xbasisi-(i)] += f*d*B(xi,p,xbasisi,knotvector)*J
    return Ke, Fe
def u_star(x,xA=0,xB=10,uA=1,uB=0):
    w1 = x-xA
    w2 = xB-x
    u_s = (w1*uB + w2*uA)/(w1+w2)
    return u_s
def assembly(K,F,Ke,Fe,elem):
    for idxx in range(len(Fe)):
        for idxy in range(len(Fe)):
            K[elem+idxx, elem+idxy] += Ke[idxx,idxy]
        F[elem+idxx] += Fe[idxx]
    return K,F
def solve(F,K):
    zero_rows = np.all(K == 0, axis=1)
    zero_cols = np.all(K == 0, axis=0)
    zero_f = zero_rows
    # Remove zero rows and columns
    K_reduced = K[~zero_rows][:, ~zero_cols]
    F_reduced = F[~zero_f]
    u = np.zeros(len(F))
    u_red = np.dot(np.linalg.inv(K_reduced),F_reduced)
    u[~zero_f] = u_red
    print("U:\n",u)
    return u
def evaluate(u,t,p,xx):
    sum = np.zeros(xx.shape)
    for i in range(0,len(u)):
        Ni = np.array([disctance(x)*B(x,p,i,t) for x in xx])
        ui = u[i]
        Ni = ui*Ni
        sum +=Ni
    #sum = np.zeros(xx.shape)
    sum+=[u_star(x) for x in xx]
    return sum
def show(u,t,p,xx,analitical = None, plotbasis=False):
    sum = np.zeros(xx.shape)
    for i in range(0,len(u)):
        Ni = np.array([disctance(x)*B(x,p,i,t) for x in xx])
        ui = u[i]
        Ni = ui*Ni
        if plotbasis: plt.plot(xx, Ni,'--g')
        sum +=Ni
    #sum = np.zeros(xx.shape)
    sum+=[u_star(x) for x in xx]
    #plt.plot(xx,[u_star(x) for x in xx])
    #analitical= np.array(analitical)-[u_star(x) for x in xx]
    plt.plot(xx,sum)
    if analitical is not None:
        plt.plot(xx,analitical)
    if not plotbasis: plt.legend(["Numerical","Analitical"])
    plt.show()
