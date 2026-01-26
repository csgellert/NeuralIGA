import matplotlib.pyplot as plt
import numpy as np
import math
from bspline import Bspline
from torch import nn
import torch
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
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate the perpendicular distance from point (px, py) to the line segment (x1, y1) -> (x2, y2)."""
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_sq == 0:  # The segment is a point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
def torch_distance_point_to_line(px, py, x1, y1, x2, y2):
   line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
   eps = torch.finfo(px.dtype).eps

   if line_length_sq == 0:
      return torch.sqrt((px - x1) ** 2 + (py - y1) ** 2 + eps)

   t = torch.clamp(
      ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq,
      0.0,
      1.0,
   )
   proj_x = x1 + t * (x2 - x1)
   proj_y = y1 + t * (y2 - y1)
   return torch.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2 + eps)


def l_shape_distance(crd):
   """Signed distance to the L-shaped domain boundary.

   Negative inside, positive outside.
   """
   x = crd[..., 0]
   y = crd[..., 1]

   corners = [
      (-1.0, 1.0),
      (-1.0, -1.0),
      (1.0, -1.0),
      (1.0, 0.0),
      (0.0, 0.0),
      (0.0, 1.0),
      (-1.0, 1.0),
   ]

   dists = [
      torch_distance_point_to_line(
         x,
         y,
         corners[i][0],
         corners[i][1],
         corners[i + 1][0],
         corners[i + 1][1],
      )
      for i in range(len(corners) - 1)
   ]
   dist = torch.min(torch.stack(dists), dim=0).values

   inside_rect1 = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 0)
   inside_rect2 = (x >= -1) & (x <= 0) & (y > 0) & (y <= 1)
   inside = inside_rect1 | inside_rect2
   sign = -torch.where(inside, -1.0, 1.0)
   sign = sign.to(dist.dtype).to(dist.device)
   return dist * sign


def smooth_min(a, b, k=1.0):
   """Smooth minimum using Rvachev R-functions concept.

   k controls smoothness: higher k = sharper transition (harder min)
   Returns smooth approximation to min(a,b) with continuous derivatives.
   Note: this formulation does *not* guarantee that zero isolines are
   preserved when one argument is exactly zero (it slightly shifts the
   zero level by ~k/2).
   """
   return (a + b - torch.sqrt((a - b)**2 + k**2)) / 2.0


def smooth_min_preserve_zero(a, b, eps=1e-12):
   """Zero-preserving smooth minimum (Rvachev R0-type).

   This keeps the property that if either input is exactly zero and the
   other is positive, the output is zero. It is continuously differentiable
   and better preserves the exact contour of distance fields.

   Args:
      a, b: tensors to combine
      eps: small stabilizer to avoid sqrt(0)
   """
   return a + b - torch.sqrt(a * a + b * b + eps)


def l_shape_distance_smooth(crd, k=0.1, preserve_zero_line=False):
   """Smooth distance to L-shaped domain boundary using R-functions.

   Uses smooth min operations instead of hard min to create a continuously
   differentiable distance function. If ``preserve_zero_line`` is True the
   zero isoline coincides with the exact contour (distance = 0 only on
   the boundary) using a zero-preserving R-function.

   Args:
      crd: coordinates tensor with shape (..., 2)
      k: smoothness parameter (used only when preserve_zero_line=False)
      preserve_zero_line: if True, use zero-preserving smooth min that
         exactly keeps the contour at distance 0

   Returns:
      Smooth signed distance. Negative inside, positive outside.
   """
   x = crd[..., 0]
   y = crd[..., 1]

   corners = [
      (-1.0, 1.0),
      (-1.0, -1.0),
      (1.0, -1.0),
      (1.0, 0.0),
      (0.0, 0.0),
      (0.0, 1.0),
      (-1.0, 1.0),
   ]

   # Compute distances to all edges
   dists = [
      torch_distance_point_to_line(
         x,
         y,
         corners[i][0],
         corners[i][1],
         corners[i + 1][0],
         corners[i + 1][1],
      )
      for i in range(len(corners) - 1)
   ]

   # Use smooth min operation instead of hard min
   # Choose zero-preserving R-function when requested
   dist = dists[0]
   if preserve_zero_line:
      for d in dists[1:]:
         dist = smooth_min_preserve_zero(dist, d)
   else:
      for d in dists[1:]:
         dist = smooth_min(dist, d, k=k)

   # Determine if point is inside using the same logic
   inside_rect1 = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 0)
   inside_rect2 = (x >= -1) & (x <= 0) & (y > 0) & (y <= 1)
   inside = inside_rect1 | inside_rect2
   sign = -torch.where(inside, -1.0, 1.0)
   sign = sign.to(dist.dtype).to(dist.device)
   return dist * sign


def dist_to_circle(crd):
   x = crd[..., 0]
   y = crd[..., 1]
   # Use a safe norm to avoid undefined gradient at (0,0) when using autograd.
   # This prevents NaNs in dx/dy during Gauss quadrature for p=3 (Gauss points include 0).
   eps = torch.finfo(x.dtype).eps
   return 1 - torch.sqrt(x**2 + y**2 + eps)

def dist_to_circle_derivative(crd):
    x = crd[0]
    y = crd[1]
    norm = np.sqrt(x**2 + y**2)
    if norm == 0:
        return np.array([0, 0])
    return -np.array([x, y]) / norm
class AnaliticalDistanceCircle(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, crd):
      return dist_to_circle(crd)

   def create_contour_plot(self, resolution=100):
      x = np.linspace(0, 1, resolution)
      y = np.linspace(0, 1, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='Distance')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Contour plot of distance function')
      plt.show()
class AnaliticalDistanceLshape(nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, crd):
      return l_shape_distance(crd)
   def create_contour_plot(self, resolution=100):
      x = np.linspace(0, 1, resolution)
      y = np.linspace(0, 1, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='Distance')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Contour plot of distance function')
      plt.show()


class AnaliticalDistanceLshape_RFunction(nn.Module):
   """Smooth L-shape distance using Rvachev R-functions.
   
   This implements a smooth (continuously differentiable) distance function
   to the L-shaped domain boundary using smooth min operations instead of
   hard minimum. This avoids sharp corners and makes the function suitable
   for neural IGA applications.
   """
   def __init__(self, smoothness=0.1, preserve_zero_line=True):
      """
      Args:
         smoothness: R-function smoothness parameter k (used when
            preserve_zero_line is False). Typical range: 0.01 - 0.5
         preserve_zero_line: if True, uses zero-preserving R-function so
            distance==0 exactly on the contour (recommended)
      """
      super().__init__()
      self.smoothness = smoothness
      self.preserve_zero_line = preserve_zero_line

   def forward(self, crd):
      return l_shape_distance_smooth(
         crd,
         k=self.smoothness,
         preserve_zero_line=self.preserve_zero_line,
      )

   def create_contour_plot(self, resolution=100):
      x = np.linspace(-1.01, 1.01, resolution)
      y = np.linspace(-1.01, 1.01, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='Smooth Distance')
      #zero contour
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      title = 'Smooth L-shape distance'
      if self.preserve_zero_line:
         title += ' (zero-preserving)'
      else:
         title += f' (k={self.smoothness})'
      plt.title(title)
      plt.show()


class AnaliticalDistanceCircle_smooth(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, crd):
      return 1-crd[...,0]**2 - crd[...,1]**2

   def create_contour_plot(self, resolution=100):
      x = np.linspace(0, 1, resolution)
      y = np.linspace(0, 1, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='Distance')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Contour plot of distance function')
      plt.show()
if __name__ == "__main__":
   analitical_model2 = AnaliticalDistanceLshape()
   model = analitical_model2
   model.create_contour_plot(resolution=100)
