import matplotlib.pyplot as plt
import numpy as np
def B(x, k, i, t, finish_end=True): #uniform B-spline Basis Functions
   # x = xi
   # k = grade
   # i = i-th basis function
   # t = knotvector
   #! finish end: at the right side if the intervall the function shall be 0 by definition,
   #! but in our case it shall be 1 but in the recursive functon call it would cause problem
   correction_required = x == t[-1] and finish_end and not t[i+k] == t[i] and t[i+k] == t[-1]
   if k == 0:
      if correction_required: 
         #TODO: By 1st order B-spline the derivative does is still 0 at the boundary 
         return 1.0 if t[i] <= x <= t[i+1] else 0.0#! if we are at the end of the intervall we have to fix 0-order elements to not to be zero
      else:
         return 1.0 if t[i] <= x < t[i+1] else 0.0 
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, finish_end=False)
   if t[i+k+1] == t[i+1]:
      c2 = 1 if correction_required else 0 #! at the right side if the intervall the function shall be 0 by definition,but in our case it shall be 1 but in the recursive functon call it would cause problem
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t,finish_end=False)
   return c1 + c2
def dBdXi(x, k, i, t):
   assert k>=1
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = k/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = k/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 - c2

def bspline(x, t, c, k):
   # x = xi
   # t = knot vector
   # c = weigths
   # k = grade
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))

def plotBsplineBasis(x, t, k,derivative = False, sum = False):
   n = len(t) - k - 1
   assert (n >= k+1)
   fig, ax = plt.subplots()
   summ = np.zeros(len(x))
   for i in range(n):
      N = [B(xx, k, i, t) for xx in x]
      summ += N
      ax.plot(x,N,'b--')
   ax.plot(t,[0 for _ in t], 'r*')
   plt.title("Basis functions")
   if derivative:
      for i in range(n):
         d = [dBdXi(xx, k, i, t) for xx in x]
         ax.plot(x,d)
   if sum:
      ax.plot(x,summ,'c-')
   plt.show()
def Surface(u,w,knotvector_u,knotvector_w,p,q,ctrlpts):
    k = len(knotvector_u)-p-1
    l = len(knotvector_w)-q-1
    sum = np.array([0,0],dtype='float64')
    for i in range(0,k):
        for j in range(0,l):
            sum += np.array(ctrlpts[j][i])*B(u,p,i,knotvector_u)*B(w,q,j,knotvector_w)
    return list(sum)
def GenerateSurface(knotvector_u,knotvector_w,p,q,ctrlpts):
      u = np.linspace(knotvector_u[0], knotvector_u[-1], 100)
      w = np.linspace(knotvector_w[0], knotvector_w[-1], 100)
      Surfacepoints = []
      for uu in u:
         row = []
         for ww in w:
            row.append(Surface(uu, ww, knotvector_u, knotvector_w, p, q, ctrlpts))
            Surfacepoints.append(row)
      return Surfacepoints
def plotGeneratedSurface(knotvector_u, knotvector_w, p, q, ctrlpts):
   surface = GenerateSurface(knotvector_u, knotvector_w, p, q, ctrlpts)
   fig, ax = plt.subplots()
   xPoints = []
   yPoints = []
   for i in surface:
      for q in i:
         xPoints.append(q[0])  
         yPoints.append(q[1])  
   ax.scatter(xPoints, yPoints)
   # Plot control points
   x = []
   y = []
   z = []
   for j in ctrlpts:
      for i in j:
         x.append(i[0])
         y.append(i[1])
   ax.scatter(x, y, c="r", marker="*")
   plt.show()
