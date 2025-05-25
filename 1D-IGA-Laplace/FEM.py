from utility import *
import numpy as np
import matplotlib.pyplot as plt
def element(xx, t, p):
    grad = len(t)-p-1
    ctrl = np.linspace(t[0],t[-1],grad)
    K = np.zeros([grad,grad])
    F = np.zeros(grad)
    dx = (xx[2]-xx[1])
    for i in range(grad):
        #if i == 0: continue # W shall be zero at Driichlet boundary 
        for j in range(grad):
            #if j == 0: continue # W shall be zero at Driichlet boundary
            #Numerical integration:
            for x in xx:
                dxdxi = sum([dBdXi(x,p,k,t)*ctrl[k] for k in range(grad)])
                if dxdxi == 0: continue
                K[i,j] += dBdXi(x,p,i,t)*dBdXi(x,p,j,t)*dx/dxdxi

            #ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx])
        for x in xx:
            dxdxi = sum([dBdXi(x,p,k,t)*ctrl[k] for k in range(grad)])
            x_orig = sum([B(x,p,k,t)*ctrl[k] for k in range(grad)])
            F[i] += B(x,p,i,t)*(6-6*x_orig+12*x_orig**2-20*x_orig**3)*dx*dxdxi
        F[i]+= 50* B(xx[-1],p,i,t)
    #print(K)
    #print(F)
    return K,F
def element_old(xx, t, p):
    grad = len(t)-p-1
    K = np.zeros([grad,grad])
    F = np.zeros(grad)
    dx = (xx[2]-xx[1])
    for i in range(grad):
        if i == 0: continue # W shall be zero at Driichlet boundary 
        for j in range(grad):
            #Numerical integration:
            for x in xx:
                K[i,j] += dBdXi(x,p,i,t)*dBdXi(x,p,j,t)*dx

            #ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx])
        for x in xx:
            F[i] += B(x,p,i,t)*(6-6*x+12*x**2-20*x**3)*dx
        F[i]+= 50* B(xx[-1],p,i,t)
    #print(K)
    #print(F)
    return K,F
def assembly():
    pass
def solve(F,K):
    u = np.dot(np.linalg.inv(K[1:,1:]),F[1:])
    print("U:\n",u)
    return u
def evaluate(u,t,p,xx):
    sum = np.zeros(xx.shape)
    for i in range(1,len(t)-p-1):
        Ni = np.array([B(x,p,i,t) for x in xx])
        ui = u[i-1]
        Ni = ui*Ni
        #plt.plot(xx, Ni,'--g')
        sum +=Ni
    sum+=1
    return sum
def evaluate_new(u,t,p,xx):
    grad = len(t)-p-1
    ctrl = np.linspace(t[0],t[-1],grad)
    sum = np.zeros(xx.shape)
    for i in range(1,len(t)-p-1):
        Ni = np.array([B(x,p,i,t) for x in xx])
        ui = u[i-1]
        Ni = ui*Ni
        #plt.plot(xx, Ni,'--g')
        sum +=Ni
    sum+=1
    return sum
def show(u,t,p,xx,analitical = None, plotbasis=False):
    grad = len(t)-p-1
    ctrl = np.linspace(t[0],t[-1],grad)
    sum = np.zeros(xx.shape)
    for i in range(1,len(t)-p-1):
        Ni = np.array([B(x,p,i,t) for x in xx])
        ui = u[i-1]
        Ni = ui*Ni
        if plotbasis: plt.plot(xx, Ni,'--g')
        sum +=Ni
    sum+=1#np.array([B(x,p,0,t) for x in xx])
    x_orig = np.zeros(xx.shape)
    for i in range(grad):
        x_orig += ctrl[i]*np.array([B(x,p,i,t) for x in xx])
    plt.plot(x_orig,sum)

    if analitical:
        plt.plot(x_orig,analitical)
    if not plotbasis: plt.legend(["Numerical","Analitical"])
    plt.show()
def show_old(u,t,p,xx,analitical = None, plotbasis=False):
    
    sum = np.zeros(xx.shape)
    for i in range(1,len(t)-p-1):
        Ni = np.array([B(x,p,i,t) for x in xx])
        ui = u[i-1]
        Ni = ui*Ni
        if plotbasis: plt.plot(xx, Ni,'--g')
        sum +=Ni
    sum+=1
    plt.plot(xx,sum)
    if analitical:
        plt.plot(xx,analitical)
    if not plotbasis: plt.legend(["Numerical","Analitical"])
    plt.show()
