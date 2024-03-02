from utility import *
import numpy as np
import matplotlib.pyplot as plt
def element(xx, t, k):
    grad = len(t)-k-1
    K = np.zeros([grad,grad])
    F = np.zeros(grad)
    for i in range(grad):
        if i == 0: continue
        for j in range(grad):
            # Ni = [B(x,k,i,t) for x in xx]
            # Nj = [B(x,k,j,t) for x in xx]
            # dNi = [dBdXi(x,k,i,t) for x in xx]
            # dNj = [dBdXi(x,k,j,t) for x in xx]
            #Numerical integration:
            for x in xx:
                K[i,j] += dBdXi(x,k,i,t)*dBdXi(x,k,j,t)*(xx[1]-xx[2])

            #ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx])
        fsdf = xx[-1]
        d = dBdXi(xx[-2],k,i,t)
        F[i] = 50* d
    return K,F
def assembly():
    pass
def solve(F,K):
    u = np.dot(np.linalg.inv(K[1:,1:]),F[1:])
    u = np.insert(u,0,0)
    return u
def show(u,t,k,xx):
    sum = np.zeros(xx.shape)
    for i in range(len(t)-k-1):
        Ni = np.array([B(x,k,i,t) for x in xx])
        ui = u[i]
        Ni = ui*Ni
        plt.plot(xx, Ni,'--g')
        sum +=Ni
    plt.plot(xx,sum)
    plt.show()
