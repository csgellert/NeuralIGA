import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Optional
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


def _point_in_convex_polygon_ccw(crd: torch.Tensor, vertices_ccw: torch.Tensor) -> torch.Tensor:
   """Return boolean mask whether points are inside a CCW convex polygon.

   Args:
      crd: tensor of shape (..., 2)
      vertices_ccw: tensor of shape (m, 2) ordered counter-clockwise

   Returns:
      inside: boolean tensor of shape (...,)
   """
   v0 = vertices_ccw
   v1 = torch.roll(vertices_ccw, shifts=-1, dims=0)
   edge = v1 - v0  # (m,2)
   rel = crd[..., None, :] - v0  # (...,m,2)
   cross = edge[:, 0] * rel[..., :, 1] - edge[:, 1] * rel[..., :, 0]  # (...,m)
   return torch.all(cross >= 0, dim=-1)


def _halfplane_distances(
   crd: torch.Tensor,
   vertices_ccw: torch.Tensor,
) -> list:
   """Signed perpendicular distances to each edge *line* of a convex polygon.

   For CCW-ordered vertices the inward normal of edge (v_i -> v_{i+1}) is
   obtained by rotating the edge direction 90° counter-clockwise: (-e_y, e_x).
   The signed distance h_i = n_i · (p - v_i) is positive inside the half-plane,
   negative outside, and exactly zero on the edge line.

   Because each h_i is a *linear* function of (x, y) it is infinitely
   differentiable everywhere — no sqrt, no eps stabiliser needed.
   """
   v0 = vertices_ccw
   v1 = torch.roll(vertices_ccw, shifts=-1, dims=0)

   dists = []
   for i in range(vertices_ccw.shape[0]):
      ex = v1[i, 0] - v0[i, 0]
      ey = v1[i, 1] - v0[i, 1]
      length = torch.sqrt(ex * ex + ey * ey)
      # Inward normal for CCW polygon: (-ey, ex) / |e|
      nx = -ey / length
      ny = ex / length
      h = nx * (crd[..., 0] - v0[i, 0]) + ny * (crd[..., 1] - v0[i, 1])
      dists.append(h)

   return dists


def convex_polygon_distance_smooth(
   crd: torch.Tensor,
   vertices_ccw: torch.Tensor,
   preserve_zero_line: bool = True,
   k: float = 0.1,
   eps: float = 1e-12,
) -> torch.Tensor:
   """Smooth signed distance-like function to a convex polygon boundary.

   The result is *distance-like* (0 on boundary, >0 inside, <0 outside) and is
   continuously differentiable for smooth-min variants.

   Notes:
   - True signed distance is not smooth at vertices; this is an approximation
     intended as a WEB/collocation weight w with usable derivatives.
   - ``preserve_zero_line=True`` uses signed half-plane distances (linear
     functions, exactly 0 on each edge line) combined with the R0 R-function
     (eps=0). This preserves the w=0 contour analytically. The only point
     where the gradient is singular is at polygon vertices (two half-planes
     simultaneously zero) — a measure-zero set that collocation grids do
     not hit in practice.
   """
   if preserve_zero_line:
      # Half-plane distances: linear → infinitely differentiable, exactly 0
      # on the edge lines.  R0 with eps=0 preserves zeros analytically.
      # The sign (positive inside, negative outside) is built into the
      # half-plane distances, so no separate inside/sign logic is needed.
      hp = _halfplane_distances(crd, vertices_ccw)
      dist = hp[0]
      for h in hp[1:]:
         dist = smooth_min_preserve_zero(dist, h, eps=0.0)
      return dist
   else:
      x = crd[..., 0]
      y = crd[..., 1]

      v0 = vertices_ccw
      v1 = torch.roll(vertices_ccw, shifts=-1, dims=0)

      dists = [
         torch_distance_point_to_line(
            x,
            y,
            v0[i, 0],
            v0[i, 1],
            v1[i, 0],
            v1[i, 1],
         )
         for i in range(vertices_ccw.shape[0])
      ]

      dist = dists[0]
      for d in dists[1:]:
         dist = smooth_min(dist, d, k=k)

      inside = _point_in_convex_polygon_ccw(crd, vertices_ccw)
      sign = torch.where(inside, torch.tensor(1.0, dtype=dist.dtype, device=dist.device), torch.tensor(-1.0, dtype=dist.dtype, device=dist.device))
      return dist * sign


def convex_polygon_sdf(
   crd: torch.Tensor,
   vertices_ccw: torch.Tensor,
   eps: float = 1e-12,
) -> torch.Tensor:
   """True signed distance function for a convex polygon.

   Computes exact Euclidean distance to the nearest edge using hard min.
   Positive inside, negative outside, zero on boundary.

   Unlike convex_polygon_distance_smooth, this uses torch.min (not R-functions),
   giving the true SDF. Non-smooth at interior ridges equidistant from two edges.
   """
   x = crd[..., 0]
   y = crd[..., 1]

   v0 = vertices_ccw
   v1 = torch.roll(vertices_ccw, shifts=-1, dims=0)

   dists = torch.stack([
      torch_distance_point_to_line(
         x, y, v0[i, 0], v0[i, 1], v1[i, 0], v1[i, 1],
      )
      for i in range(vertices_ccw.shape[0])
   ], dim=-1)

   dist = torch.min(dists, dim=-1).values

   inside = _point_in_convex_polygon_ccw(crd, vertices_ccw)
   sign = torch.where(inside,
                      torch.tensor(1.0, dtype=dist.dtype, device=dist.device),
                      torch.tensor(-1.0, dtype=dist.dtype, device=dist.device))
   return dist * sign


def hollig_transform(d: torch.Tensor, delta: float = 0.1, gamma: float = 3.0) -> torch.Tensor:
   """Höllig weight function applied to a signed distance field.

   w(d) = 1 - clamp(1 - d/delta, min=0)^gamma

   Maps: d=0 -> w=0 (boundary), d>=delta -> w=1 (deep inside).
   For d<0 (outside), w<0. Smoothness class C^{gamma-1}.
   Same formula as mesh.hollig_weight.
   """
   term = 1.0 - d / delta
   term = torch.clamp(term, min=0.0)
   return 1.0 - torch.pow(term, gamma)


def regular_polygon_vertices_np(
   n_sides: int,
   radius: float = 0.9,
   center: tuple = (0.0, 0.0),
   rotation: float = 0.0,
) -> np.ndarray:
   """Create CCW vertices for a regular polygon (NumPy)."""
   cx, cy = float(center[0]), float(center[1])
   angles = rotation + np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
   x = cx + radius * np.cos(angles)
   y = cy + radius * np.sin(angles)
   return np.stack([x, y], axis=1)


def ensure_vertices_ccw_np(vertices: np.ndarray) -> np.ndarray:
   """Return vertices ordered CCW (NumPy).

   Uses the signed polygon area: positive => CCW.
   """
   v = np.asarray(vertices, dtype=np.float64)
   if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
      raise ValueError("vertices must have shape (m,2) with m>=3")
   x = v[:, 0]
   y = v[:, 1]
   x1 = np.roll(x, -1)
   y1 = np.roll(y, -1)
   area2 = np.sum(x * y1 - y * x1)
   if area2 < 0:
      return v[::-1].copy()
   return v


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
      x = np.linspace(-1.005, 1.005, resolution)
      y = np.linspace(-1.005, 1.005, resolution)
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
      x = np.linspace(-1.005, 1.005, resolution)
      y = np.linspace(-1.005, 1.005, resolution)
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

class AnaliticalDistance_CASE8(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, crd):
      e = 1-(crd[...,0]**2)/16 -(crd[...,1]**2)/9
      k = crd[...,0]**2+1.5*crd[...,0]+crd[...,1]**2-crd[...,1]-3/16
      return e*k

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


class AnaliticalDistanceTriangle_RFunction(nn.Module):
   """Smooth triangle weight function (convex polygon) on [-1,1]^2."""

   def __init__(
      self,
      smoothness: float = 0.1,
      preserve_zero_line: bool = True,
      vertices: Optional[np.ndarray] = None,
   ):
      super().__init__()
      self.smoothness = float(smoothness)
      self.preserve_zero_line = bool(preserve_zero_line)

      # Default: a reasonably sized CCW triangle inside [-1,1]^2.
      if vertices is None:
         vertices = np.array(
            [
               [-0.85, -0.60],
               [0.90, -0.55],
               [0.05, 0.92],
            ],
            dtype=np.float64,
         )
      self._vertices_np = ensure_vertices_ccw_np(np.asarray(vertices, dtype=np.float64))

   def forward(self, crd: torch.Tensor) -> torch.Tensor:
      verts = crd.new_tensor(self._vertices_np)
      return convex_polygon_distance_smooth(
         crd,
         verts,
         preserve_zero_line=self.preserve_zero_line,
         k=self.smoothness,
      )

   def create_contour_plot(self, resolution=200):
      x = np.linspace(-1.01, 1.01, resolution)
      y = np.linspace(-1.01, 1.01, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      title = 'Triangle weight function'
      title += ' (zero-preserving)' if self.preserve_zero_line else f' (k={self.smoothness})'
      plt.title(title)
      plt.show()


class AnaliticalDistancePentagon_RFunction(nn.Module):
   """Smooth regular pentagon weight function (convex polygon) on [-1,1]^2."""

   def __init__(
      self,
      smoothness: float = 0.1,
      preserve_zero_line: bool = True,
      radius: float = 0.9,
      rotation: float = np.pi / 2,
      center: tuple = (0.0, 0.0),
   ):
      super().__init__()
      self.smoothness = float(smoothness)
      self.preserve_zero_line = bool(preserve_zero_line)
      self._vertices_np = ensure_vertices_ccw_np(regular_polygon_vertices_np(
         5,
         radius=float(radius),
         center=center,
         rotation=float(rotation),
      ).astype(np.float64))

   def forward(self, crd: torch.Tensor) -> torch.Tensor:
      verts = crd.new_tensor(self._vertices_np)
      return convex_polygon_distance_smooth(
         crd,
         verts,
         preserve_zero_line=self.preserve_zero_line,
         k=self.smoothness,
      )

   def create_contour_plot(self, resolution=200):
      x = np.linspace(-1.01, 1.01, resolution)
      y = np.linspace(-1.01, 1.01, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      title = 'Pentagon weight function'
      title += ' (zero-preserving)' if self.preserve_zero_line else f' (k={self.smoothness})'
      plt.title(title)
      plt.show()


class AnaliticalDistanceTriangle_SDF_Hollig(nn.Module):
   """True SDF for a triangle, post-processed with Höllig weight function.

   Computes exact signed distance to the triangle boundary (hard min of
   per-edge distances), then applies the Höllig transformation
   w(d) = 1 - max(0, 1 - d/delta)^gamma.
   The resulting weight is C^{gamma-1} smooth near the boundary and
   equals 1 deep inside the domain.
   """

   def __init__(
      self,
      vertices: Optional[np.ndarray] = None,
      delta: float = 0.1,
      gamma: float = 3.0,
   ):
      super().__init__()
      self.delta = float(delta)
      self.gamma = float(gamma)
      if vertices is None:
         vertices = np.array(
            [[-0.85, -0.60], [0.90, -0.55], [0.05, 0.92]], dtype=np.float64,
         )
      self._vertices_np = ensure_vertices_ccw_np(np.asarray(vertices, dtype=np.float64))

   def forward(self, crd: torch.Tensor) -> torch.Tensor:
      verts = crd.new_tensor(self._vertices_np)
      sdf = convex_polygon_sdf(crd, verts)
      return hollig_transform(sdf, delta=self.delta, gamma=self.gamma)

   def create_contour_plot(self, resolution=200):
      x = np.linspace(-1.01, 1.01, resolution)
      y = np.linspace(-1.01, 1.01, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w (Höllig)')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title(f'Triangle SDF + Höllig (δ={self.delta}, γ={self.gamma})')
      plt.show()


class AnaliticalDistancePentagon_SDF_Hollig(nn.Module):
   """True SDF for a regular pentagon, post-processed with Höllig weight function."""

   def __init__(
      self,
      delta: float = 0.1,
      gamma: float = 3.0,
      radius: float = 0.9,
      rotation: float = np.pi / 2,
      center: tuple = (0.0, 0.0),
   ):
      super().__init__()
      self.delta = float(delta)
      self.gamma = float(gamma)
      self._vertices_np = ensure_vertices_ccw_np(regular_polygon_vertices_np(
         5, radius=float(radius), center=center, rotation=float(rotation),
      ).astype(np.float64))

   def forward(self, crd: torch.Tensor) -> torch.Tensor:
      verts = crd.new_tensor(self._vertices_np)
      sdf = convex_polygon_sdf(crd, verts)
      return hollig_transform(sdf, delta=self.delta, gamma=self.gamma)

   def create_contour_plot(self, resolution=200):
      x = np.linspace(-1.01, 1.01, resolution)
      y = np.linspace(-1.01, 1.01, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w (Höllig)')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title(f'Pentagon SDF + Höllig (δ={self.delta}, γ={self.gamma})')
      plt.show()


class AnaliticalDistanceCircle_smooth(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, crd):
      return 1-crd[...,0]**2 - crd[...,1]**2

   def create_contour_plot(self, resolution=100):
      x = np.linspace(-1.005, 1.005, resolution)
      y = np.linspace(-1.005, 1.005, resolution)
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


class AnaliticalDistanceDisc_WEB(nn.Module):
   """Analytic WEB-spline 'disc' weight function on [0,1]^2.

   Matches collocation_WEB.DiscWeightFunction / example_exact_solution_disc:
      w(x,y) = 1 - (2x-1)^2 - (2y-1)^2
   Domain: {(x,y) in [0,1]^2 : w(x,y) > 0} (disc radius 0.5 centered at (0.5,0.5)).
   """

   def __init__(self):
      super().__init__()

   def forward(self, crd):
      x = crd[..., 0]
      y = crd[..., 1]
      return 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2

   def create_contour_plot(self, resolution=200):
      x = np.linspace(0.0, 1.0, resolution)
      y = np.linspace(0.0, 1.0, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('WEB disc weight function w (zero contour shown)')
      plt.show()
class AnaliticalDistanceEllipsePentagon_RFunction(nn.Module):
   """Five-sided domain bounded by ellipse arcs, using algebraic distance.

   Each "side" is defined by an ellipse. The domain is the intersection
   of the interiors of all five ellipses. The weight function uses the
   algebraic distance (implicit function value) for each ellipse, combined
   via the R0 R-function (smooth_min_preserve_zero).

   Algebraic distance for ellipse k:
      F_k(x,y) = 1 - [(delta . t_k)^2 / a_k^2 + (delta . r_k)^2 / b_k^2]

   where delta = (x - cx_k, y - cy_k), t_k is the tangential direction,
   r_k is the radial direction, a_k is the tangential semi-axis, and
   b_k is the radial semi-axis.

   Parameters
   ----------
   n_sides : int
       Number of sides (default 5).
   center_distance : float
       Distance from origin to each ellipse center.
   semi_axis_tangential : float or array-like
       Semi-axis along the tangential direction (parallel to side).
   semi_axis_radial : float or array-like
       Semi-axis along the radial direction (perpendicular to side).
   rotation_offset : float
       Rotation of the entire shape (radians). Default pi/2 (first
       center at top).
   preserve_zero_line : bool
       Whether to use the zero-preserving R-function combination.
   eps : float
       Smoothing parameter for the R-function.
   """

   def __init__(
      self,
      n_sides: int = 5,
      center_distance: float = 1.5,
      semi_axis_tangential: float = 1.2,
      semi_axis_radial: float = 0.85,
      rotation_offset: float = np.pi / 2,
      preserve_zero_line: bool = True,
      eps: float = 1e-6,
   ):
      super().__init__()
      self.n_sides = n_sides
      self.preserve_zero_line = bool(preserve_zero_line)
      self.eps = float(eps)

      angles = rotation_offset + 2 * np.pi * np.arange(n_sides) / n_sides

      if np.isscalar(semi_axis_tangential):
         semi_axis_tangential = np.full(n_sides, float(semi_axis_tangential))
      if np.isscalar(semi_axis_radial):
         semi_axis_radial = np.full(n_sides, float(semi_axis_radial))

      self._centers_np = np.stack([
         center_distance * np.cos(angles),
         center_distance * np.sin(angles),
      ], axis=-1).astype(np.float64)  # (n_sides, 2)

      self._angles_np = np.asarray(angles, dtype=np.float64)
      self._semi_a_np = np.asarray(semi_axis_tangential, dtype=np.float64)
      self._semi_b_np = np.asarray(semi_axis_radial, dtype=np.float64)

      # Pre-compute boundary path for visualization
      self._boundary_pts = self._compute_boundary_path(500)

   def _compute_boundary_path(self, n_pts: int = 500) -> np.ndarray:
      """Sample the boundary (w approx 0) via polar bisection."""
      theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
      r_boundary = np.zeros(n_pts)
      for i in range(n_pts):
         cos_th = np.cos(theta[i])
         sin_th = np.sin(theta[i])
         r_lo, r_hi = 0.0, 3.0
         for _ in range(60):
            r_mid = 0.5 * (r_lo + r_hi)
            x, y = r_mid * cos_th, r_mid * sin_th
            min_F = self._min_algebraic_distance_scalar(x, y)
            if min_F > 0:
               r_lo = r_mid
            else:
               r_hi = r_mid
         r_boundary[i] = 0.5 * (r_lo + r_hi)
      pts = np.stack([r_boundary * np.cos(theta),
                      r_boundary * np.sin(theta)], axis=-1)
      return pts

   def _min_algebraic_distance_scalar(self, x: float, y: float) -> float:
      """Min algebraic distance at a single point (numpy, for bisection)."""
      min_F = float('inf')
      for k in range(self.n_sides):
         alpha = self._angles_np[k]
         cx, cy = self._centers_np[k]
         a, b = self._semi_a_np[k], self._semi_b_np[k]
         dx, dy = x - cx, y - cy
         sa, ca = np.sin(alpha), np.cos(alpha)
         pt = -sa * dx + ca * dy
         pr = ca * dx + sa * dy
         F = 1.0 - pt ** 2 / a ** 2 - pr ** 2 / b ** 2
         min_F = min(min_F, F)
      return min_F

   def forward(self, crd: torch.Tensor) -> torch.Tensor:
      dists = self._algebraic_distances_torch(crd)
      if self.preserve_zero_line:
         result = dists[0]
         for i in range(1, len(dists)):
            result = smooth_min_preserve_zero(result, dists[i], eps=self.eps)
      else:
         result = dists[0]
         for i in range(1, len(dists)):
            result = smooth_min(result, dists[i], k=self.eps)
      return result

   def _algebraic_distances_torch(self, crd: torch.Tensor):
      """Compute algebraic distance for each ellipse (torch, differentiable)."""
      centers = crd.new_tensor(self._centers_np)
      angles = crd.new_tensor(self._angles_np)
      semi_a = crd.new_tensor(self._semi_a_np)
      semi_b = crd.new_tensor(self._semi_b_np)

      dists = []
      for k in range(self.n_sides):
         dx = crd[..., 0] - centers[k, 0]
         dy = crd[..., 1] - centers[k, 1]
         cos_a = torch.cos(angles[k])
         sin_a = torch.sin(angles[k])

         proj_t = -sin_a * dx + cos_a * dy  # tangential
         proj_r = cos_a * dx + sin_a * dy   # radial

         F_k = 1.0 - (proj_t ** 2 / semi_a[k] ** 2
                       + proj_r ** 2 / semi_b[k] ** 2)
         dists.append(F_k)
      return dists

   def algebraic_distances_numpy(self, x, y):
      """Compute individual algebraic distances as numpy arrays."""
      x = np.asarray(x, dtype=np.float64)
      y = np.asarray(y, dtype=np.float64)
      dists = []
      for k in range(self.n_sides):
         alpha = self._angles_np[k]
         cx, cy = self._centers_np[k]
         a, b = self._semi_a_np[k], self._semi_b_np[k]
         dx = x - cx
         dy = y - cy
         sa, ca = np.sin(alpha), np.cos(alpha)
         pt = -sa * dx + ca * dy
         pr = ca * dx + sa * dy
         F = 1.0 - pt ** 2 / a ** 2 - pr ** 2 / b ** 2
         dists.append(F)
      return dists

   def min_algebraic_distance_numpy(self, x, y):
      """Min algebraic distance over all ellipses (numpy)."""
      dists = self.algebraic_distances_numpy(x, y)
      return np.min(np.stack(dists, axis=-1), axis=-1)

   def create_contour_plot(self, resolution=200):
      x = np.linspace(-1.2, 1.2, resolution)
      y = np.linspace(-1.2, 1.2, resolution)
      X, Y = np.meshgrid(x, y)
      crd = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
      with torch.no_grad():
         Z = self.forward(crd).cpu().numpy()
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.colorbar(label='w (algebraic distance)')
      plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
      bp = self._boundary_pts
      plt.plot(np.append(bp[:, 0], bp[0, 0]),
               np.append(bp[:, 1], bp[0, 1]),
               'r--', linewidth=1, label='boundary')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Ellipse pentagon (algebraic distance, R-function)')
      plt.legend()
      plt.gca().set_aspect('equal')
      plt.show()


if __name__ == "__main__":
   analitical_model2 = AnaliticalDistanceLshape()
   model = analitical_model2
   model.create_contour_plot(resolution=100)
