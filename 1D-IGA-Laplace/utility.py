import matplotlib.pyplot as plt
def B(x, k, i, t, finish_end=True): #uniform B-spline Basis Functions
   # x = xi
   # k = grade
   # i = i-th basis function
   # t = knotvector
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t, finish_end=False)
   if t[i+k+1] == t[i+1]:
      c2 = 1 if x == t[-1] and finish_end else 0
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

def plotBsplineBasis(x, t, k):
   n = len(t) - k - 1
   assert (n >= k+1)
   fig, ax = plt.subplots()
   for i in range(n):
      N = [B(xx, k, i, t) for xx in x]
      ax.plot(x,N,'b--')
   ax.plot(t,[0 for _ in t], 'r*')
   plt.title("Basis functions")
   plt.show()
"""
def get_gauss_uniform_spline(order):
   # https://pdf.sciencedirectassets.com/271610/1-s2.0-S0377042720X00021/1-s2.0-S0377042719306314/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCFKe8kAr0YYf9ZsCr%2Fkoapk46LQU04V8jq%2F1jG9VKmYgIhAI8QfUzLf%2Bn923Fx2CeU2oOE%2F3%2BxaZJ4Mjj2vrrlp6FMKrsFCJD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzBTtqH1RelTAqzvEEqjwV9rSvym%2Bpo8KdfNTCI%2FqyG1MMtVx8vlJ3hg6BuQcdV5TWC4EF12qsPpsbumdEeTIEBAURWI%2FdtjCSPdMEjzc%2FxC2v0lKNi%2BQw2hfmy4DTHuVGE7cdkr1QLL%2FP%2BHW6nI9kQ7Z9ooQ7kjlmDg9AXNVWGqae5M6Nms1j9jZkZ0BvVZt4KWwNDH6XlOLkW2oKz%2FeXShDPfBLZXuKF9emozmbDE%2Ffqr%2Fhe8MB%2B0pqH8CHrv67f4CkLj5ogMWXLDg5wKUXGGf3vkZK5GIRHt9L1xF4gr%2FjIuUoFbm25d4cb%2BKS2ry5kd0XiYyTj%2FljO2k8nH%2FbPqePO7%2Byb0TT%2BhcTqJv9dceTP%2FrhFTHryTTgkx8NHoHStJdjvSWfXHJ0FefRTSSffvDKXlZVSM%2BYaFyaVdxcbxWyzLN%2F7%2BXV5Q5WjZM0X0ohjURrknxLhVa676DIOl3Ie91CYl%2Bk2mV3Bys5xzadDfvy0PYEcZ%2F2Io6o5hWmZiqOi8YQY8oB%2Fse69hAKqdUz%2BEbc9kgASjwzaht4gZ9N8%2FAgwAgOGL4jAw7D5wTZ94lF6dYTRvW9liAU6S7Rr0CJ9TvaTeyypSiDcqLvBDIhu%2Fszm65%2BDmcdc9RzNNuW9NxyZf9qjba137rGOiKS1BZaadnLyXA1A%2Bdj35n0u7N%2Fwbdt4q%2FucjOT4dzZhckis6e96AqUMSLY0a0kJFCftW4j%2B1%2FeGZSOAqqLpz01bVfqDlZCWuRXBZyV9JZdsAU4wBzuKgr%2BVRWYjr9xaSg30uljDGEu1dAWyOFN8UptKeM7hfc9ZksCjsVmJ2JpmUWVGsOLE8H395yHWqNFjN1Tq0xMFXBZzZye%2F6hrBQO0kJOddVawru15DDcSJdilXEUWqCMI%2FWnK8GOrAB5hcVSw7BaBGg1%2FLbPEiaOfYV9qLq78mnpF%2Bno%2B%2FVxsfYYXSvk4DhtTF8ylZHx70ZdnfkZv1RchJwABduDZF%2FicO7BttPYwJH2%2FlA%2FMpCCXb%2Fa6G7wylxeoxzLmpX7Ad5Ryp7Pv4rih5IpMhOdfy8EwPe5erUfzgXlARrWKpVSscC9IwIv3U3ybsRQ2lfBJxObmHIhLJBw4U%2BFqJttpMua4ELpEu6M8k3v5ILpiK%2BICY%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240305T144624Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYSHDBZID2%2F20240305%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1d48578b643ef72b7cefcfa3c0957bebb5c44dca6a6fbbe5bb23b2c99e005f0e&hash=9b32934866fb22fa54003cf804c1dc002f1384a5bb38a5884b59551107af394d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0377042719306314&tid=spdf-c01fd381-2bd9-4c84-907f-904a170aab70&sid=2d86cd0665166546299b8d5236353e0ccd64gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=12175a51090106565556&rr=85fae5319836733d&cc=hu
   if order == 2: #quadratic case
      tau = [0.71241440095955149482, 1.5]
      omega = [0.20151829499655592436, 0.59696341000688815128]
   elif order == 3: #cubic element:
      tau = [0.72289886179270511319,1.58789880583487289415]
      omega = [0.55950733567808927174,0.44404926643219107283]
   return tau, omega
def get_gauss_points_regular_FEM(nGp=2):
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
"""

if __name__ == "__main__":
   k = 3
   
   import numpy as np
   fig, ax = plt.subplots()
   xx = np.linspace(1.5, 4.5, 500)
   xx2 = np.linspace(1, 4, 500)
   t = [ 1,1,1,1, 2, 3, 4,4,4,4]
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
      ax.plot(t,[0 for _ in t],"r*")
      #ax.plot(xx2[1:], diff[1:])
   plt.show()