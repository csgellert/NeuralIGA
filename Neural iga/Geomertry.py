import matplotlib.pyplot as plt
import numpy as np
import math
from bspline import Bspline

bsp=None
dbps = None
def init_spl(x,k,i,t):
   global bsp
   global dbps
   bsp = Bspline(t,k)
   dbps = bsp.diff(1)
   [bsp._Bspline__basis(j, bsp.p) for j in x] 
def B(x, k, i, t, finish_end=True):
   return bsp(x)[i]
def dBdXi(x, k, i, t):
   return dbps(x)[i]

def B_cdB(x, k, i, t, finish_end=True): #uniform B-spline Basis Functions
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
def dBdXi_cdB(x, k, i, t):
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

def B_gem(x, k, i, t, finish_end=True):
    # x: Can be a scalar or a NumPy array
    # k: Grade
    # i: i-th basis function
    # t: Knot vector
    # finish_end: As described in the original code

    # Handle scalar or array input for x
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    correction_required = x == t[-1] and finish_end and not t[i+k] == t[i] and t[i+k] == t[-1]

    if k == 0:
        if correction_required:
            return np.where((t[i] <= x) * (x <= t[i+1]), 1.0, 0.0)
        else:
            return np.where((t[i] <= x) * (x < t[i+1]), 1.0, 0.0)
    c1 = np.where(t[i+k] == t[i],0.0,(x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, finish_end=False))
    c2 = np.where(t[i+k+1] == t[i+1],1 if correction_required else 0,(t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, finish_end=False))
    return c1 + c2
def dBdXi_gem(x, k, i, t):
    """
    Calculates the derivative of the B-spline basis function dB(i, k, x) with respect to xi.

    Args:
        x: A NumPy array of shape (n,) representing the evaluation points.
        k: The order of the B-spline basis function.
        i: The index of the control point associated with the basis function.
        t: A NumPy array of shape (m,) representing the knot vector.

    Returns:
        A NumPy array of shape (n,) containing the derivative values at the evaluation points.
    """

    assert k >= 1

    # Calculate the derivative using vectorized operations
    c1 = np.where(t[i+k] == t[i], 0.0, k / (t[i+k] - t[i]) * B(x, k-1, i, t))
    c2 = np.where(t[i+k+1] == t[i+1], 0.0, k / (t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t))

    return c1 - c2
def cubicBspline(x, k, i, t):
   coeff = np.array([[-1,3,-3,1],[3, -6,3, 0],[-3,0,3,0],[1,4,1,0]])
   f = 1/6
   t = [x*x*x,x*x,x,1]
   basises =  f*np.dot(t,coeff)
   return basises[i]
   #!Wrong
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

if __name__ == "__main__":
   #print("2D - Immersed - FEM.py")
   x=np.linspace(-1,1,100)
   knt = [-1,-1,-1,-1,0,1,1,1,1]
   p=3
   bspe = Bspline(knt,p)
   vec = [-0.25 , 0.25]
   import bspline.splinelab as splinelab
   tmp = bspe.collmat(vec)
   print(tmp)
   tmp2 = bspe.collmat([0,0.4,0.5])
   print(tmp2)
   d = bspe.diff(1)
   print(d(0)[1])
   init_spl(x,p,None,knt)
   bspe.plot()
   import time
   strart = time.time()
   for i in range(10000):
      tmp = B(0.25,3,2,knt)
   print("sajat:",time.time()-strart)
   strart = time.time()
   for i in range(1000):
      tmp = bspe(0.25)[3]
   print("sajat:",time.time()-strart)
   print(bsp(-1))
   #plt.plot(x,bsp(x))
   plt.show()
   #plotBsplineBasis(x,knt,p)
