from utility import *
import numpy as np
import matplotlib.pyplot as plt
def element(xx, t, k):
    grad = len(t)-k-1
    K = np.zeros([grad,grad])
    F = np.zeros(grad)
    dx = (xx[2]-xx[1])
    for i in range(grad):
        if i == 0: continue
        for j in range(grad):
            # Ni = [B(x,k,i,t) for x in xx]
            # Nj = [B(x,k,j,t) for x in xx]
            # dNi = [dBdXi(x,k,i,t) for x in xx]
            # dNj = [dBdXi(x,k,j,t) for x in xx]
            #Numerical integration:
            for x in xx:
                K[i,j] += dBdXi(x,k,i,t)*dBdXi(x,k,j,t)*dx

            #ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx])
        fsdf = xx[-1]
        d = dBdXi(xx[-2],k,i,t)

        for x in xx:
            F[i] += B(x,k,i,t)*(6-6*x+12*x**2-20*x**3)*dx
        F[i]+= 50* B(xx[-2],k,i,t)
    print(K)
    print(F)
    return K,F
def assembly():
    pass
def solve(F,K):
    u = np.dot(np.linalg.inv(K[1:,1:]),F[1:])
    print("U:\n",u)
    return u
def show(u,t,k,xx,analitical = None):
    sum = np.zeros(xx.shape)
    for i in range(1,len(t)-k-1):
        Ni = np.array([B(x,k,i,t) for x in xx])
        ui = u[i-1]
        Ni = ui*Ni
        plt.plot(xx, Ni,'--g')
        sum +=Ni
    sum+=1
    plt.plot(xx,sum)
    if analitical:
        plt.plot(xx,analitical)
    plt.show()
