import numpy as np
import matplotlib.pyplot as plt

x = np.array([[-0.5, -2, 0], [1,1,1], [2,2,2]])
y = np.array([[2, 1,0], [2,0,1], [2,1,1]])
z = np.array([[1, -1,2], [0,-0.5,2], [0.5,1,2]])

#Number of cells each direction
uCells = 12
wCells  =10

#Dependant VARIABLES
#Total numer of Control Points in U
uPTS = np.size(x,0)
wPTS = np.size(x,1)

#total number of subdivision
n = uPTS -1
m = wPTS -1
#parametric variable
u = np.linspace(0,1,uCells)
w = np.linspace(0,1,wCells)

#Bernstein basis polinomial
b = []
d = []

#Initailized  Empty Matrix for X,Y,z Bezier Curve
xBezier = np.zeros((uCells,wCells))
yBezier = np.zeros((uCells,wCells))
zBezier = np.zeros((uCells,wCells))

#binomial coefficient
def Ni(n,i):
    return np.math.factorial(n)/(np.math.factorial(i)*np.math.factorial(n-i))

def Mj(m,j):
    return np.math.factorial(m)/(np.math.factorial(j)*np.math.factorial(m-j))

#Brnstein Basis polinomial
def J(n,i,u):
    return np.matrix(Ni(n,i)*(u**i)*(1-u)**(n-i))
def K(m,j,w):
    return np.matrix(Mj(m,j)*(w**j)*(1-w)**(m-j))

#MAIN LOOP
for i in range(0,uPTS):

    for j in range(0,wPTS):
        b.append(J(n,i,u))
        d.append(K(m,j,w))

        #Transpose J array
        Jt = J(n,i,u).transpose()

        #Bezier Curve calculatino
        xBezier = Jt*K(m,j,w)*x[i,j] + xBezier
        yBezier = Jt*K(m,j,w)*y[i,j] + yBezier
        zBezier = Jt*K(m,j,w)*z[i,j] + zBezier
    
#Plotting
# plt.figure()
# plt.subplot(121)
# for line in b:
#     plt.plot(u,line.transpose())
# for line in d:
#     plt.plot(w,line.transpose())
# plt.show()
        
#Bezier surface
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(xBezier,yBezier,zBezier)
ax.scatter(x,y,z, edgecolors='face')
plt.show()