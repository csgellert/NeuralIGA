import matplotlib.pyplot as plt
import math
def B(x, k, i, t, finish_end=True): #uniform B-spline Basis Functions
   # x = xi
   # k = grade
   # i = i-th basis function
   # t = knotvector
   correction_required = x == t[-1] and finish_end and not t[i+k] == t[i] and t[i+k] == t[-1]
   if k == 0:
      if correction_required: 
         #TODO The conditions shall be given proprerly: 
         return 1.0 if t[i] <= x <= t[i+1] else 0.0#! if we are at the end of the intervall we have to fix 0-order elements to not to be zero
      else:
         return 1.0 if t[i] <= x < t[i+1] else 0.0 
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, finish_end=False)
   if t[i+k+1] == t[i+1]:
      #TODO: and i<=len(t)-k-1 ????
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
def d2BdXi2(x,k,i,t):
   assert k>=2
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = k/(t[i+k] - t[i]) * dBdXi(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = k/(t[i+k+1] - t[i+1]) * dBdXi(x, k-1, i+1, t)
   return c1 - c2

def bspline(x, t, c, k):
   # x = xi
   # t = knot vector
   # c = weigths
   # k = grade
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))

def plotBsplineBasis(x, t, k,weighted = False,plotDerivatives=True):
   n = len(t) - k - 1
   assert (n >= k+1)
   fig, ax = plt.subplots()
   for i in range(n):
      if weighted:
         N = [disctance(xx)*B(xx, k, i, t) for xx in x]
      else:
         N = [B(xx, k, i, t) for xx in x]
         if plotDerivatives:
            dN = [dBdXi(xx, k, i, t) for xx in x]
            ddN = [d2BdXi2(xx, k, i, t) for xx in x]
            ax.plot(x,dN,'g-')
            ax.plot(x,ddN,'r-')
      ax.plot(x,N,'b--')
   
   ax.plot(t,[0 for _ in t], 'r*')
   plt.title("Basis functions")
   plt.show()

def disctance(x,xA=0,xB = 10):
   assert xB>xA
   w1 = x-xA
   w2 = xB-x
   w = w1+w2-math.sqrt(w1**2 + w2**2)
   return w
def distanceDerivative(x,xA=0,xB = 10):
   assert xB>xA
   w1 = x-xA
   w2 = xB-x
   dw = -1/math.sqrt(w1**2 + w2**2)*(2*x - xB-xA)
   return dw
def distance2ndDerivative(x,xA=0,xB=10):
   assert xB>xA
   w1 = x-xA
   w2 = xB-x
   d2w = (1/(math.sqrt(w1**2 + w2**2))**3 *(2*x - xB-xA)**2 - 1/math.sqrt(w1**2 + w2**2)*2)
   return d2w
if __name__ == "__main__":
   import numpy as np
   
   x = np.linspace(0, 10, 500)
   d = [disctance(xx) for xx in x]
   dd = [distanceDerivative(xx) for xx in x]
   ddd = [distance2ndDerivative(xx) for xx in x]
   plotBsplineBasis(x,[0,0,0,0,2.5,5,7.5,10,10,10,10],3)
   fig, ax = plt.subplots()
   ax.plot(x,d,"b-")
   ax.plot(x,dd,"g-")
   ax.plot(x,ddd,"r-")
   print(ddd[0],ddd[-1])
   plt.show()
