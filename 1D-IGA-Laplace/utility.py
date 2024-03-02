def B(x, k, i, t): #uniform B-spline Basis Functions
   # x = xi
   # k = grade
   # i = i-th basis function
   # t = knotvector
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
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
def get_gauss_points(nGp=2):
   if nGp == 1:
      g1 = 0.0
      w1 = 2.0
      
      xi = g1
      w  = w1
   elif nGp == 2:
      g1 = 0.577350269189626; 
      w1 = 1.0
      
      xi = [-g1, g1]
      w  = [w1, w1]
   elif nGp == 3:
      g1 = 0.774596669241483
      g2 = 0
      w1 = 0.555555555555555
      w2 = 0.888888888888888
      
      xi = [-g1, g2, g1]
      w  = [ w1, w2, w1]
   elif nGp == 4:
      g1 = 0.861136311594953
      g2 = 0.339981043584856
      w1 = 0.347854845137454
      w2 = 0.652145154862546
      
      xi = [-g1, -g2, g2, g1]
      w  = [ w1, w2, w2, w1]
   else:
      print('Number of integration points not implemented!')
      return xi, w


if __name__ == "__main__":
   k = 2
   import matplotlib.pyplot as plt
   import numpy as np
   fig, ax = plt.subplots()
   xx = np.linspace(1.5, 4.5, 500)
   xx2 = np.linspace(0, 10, 500)
   t = [ 1,1,1, 2, 3, 4,5,6,7,8,9,10,10,10]
   t2 = [4,0, 1, 2, 3, 4, 5, 6,6]
   c = [1,1,1,1,1,1,1,1,1,1,1,1]
   #ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
   ax.plot(xx2, [bspline(x, t, c ,k) for x in xx2], 'g-', lw=3, label='naive')
   #ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
   ax.grid(True)
   ax.legend(loc='best')

   for i in range(len(t)-k-1):
      #if i == 0 or i== len(t)-k-1-1: continue
      #Ni = [B(x,k,i,t) for x in xx]
      Ni2 = [B(x,k,i,t) for x in xx2]
      #diff = [(Ni2[idx]-Ni2[idx-1])/(xx2[1]-xx2[0]) for idx,x in enumerate(Ni2)]
      #ax.plot(xx, Ni)2
      ax.plot(xx2, Ni2)
      ax.plot(xx2, [dBdXi(x,k,i,t) for x in xx2])
      #ax.plot(xx2[1:], diff[1:])
   plt.show()