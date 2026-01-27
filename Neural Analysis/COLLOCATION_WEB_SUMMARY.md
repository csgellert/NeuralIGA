# WEB-Spline Collocation Method - Technical Documentation

## Executive Summary

The **WEB-Spline Collocation Method** (Weighted Extended B-Splines) is a meshfree numerical method for solving the Poisson equation on implicitly defined domains. It combines classical B-spline approximation with Lagrange extension to handle boundaries without requiring explicit mesh generation or boundary-fitted grids.

**Key Features:**
- Solves: $-\Delta u = f$ on implicitly defined domain $D: w(x,y) > 0$
- Implicit boundary: $u = 0$ on $\partial D$ (enforced by weight function vanishing)
- Meshfree: No explicit mesh, works on regular Cartesian grid
- Neural compatibility: Can use neural network SDFs as weight functions
- Based on: Höllig, Reif, Wipper (2001) - Weighted Extended B-Spline Approximation

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Core Components](#4-core-components)
5. [Implementation Details](#5-implementation-details)
6. [Usage Guide](#6-usage-guide)
7. [References](#7-references)

---

## 1. Problem Formulation

### 1.1 Boundary Value Problem

**Strong Form:**
$$\begin{cases}
-\Delta u = f & \text{in } D \\
u = 0 & \text{on } \partial D
\end{cases}$$

where:
- $D = \{(x,y) : w(x,y) > 0\}$ - implicitly defined domain
- $w(x,y)$ - **weight function** (e.g., signed distance function)
- $f(x,y)$ - given load/source function
- $u(x,y)$ - unknown solution

### 1.2 Implicit Domain Definition

The domain $D$ is defined via a weight function $w(x,y)$:
$$D = \{(x,y) \in [0,1]^2 : w(x,y) > 0\}$$

**Examples:**
- **Circle:** $w(x,y) = 1 - (2x-1)^2 - (2y-1)^2$ → unit disc centered at $(0.5, 0.5)$
- **L-shape:** $w(x,y) = \text{SDF}_L(x,y)$ → signed distance to L-shaped boundary
- **Neural SDF:** $w(x,y) = \text{NeuralNet}(x,y)$ → learned implicit geometry

The boundary $\partial D$ is the zero level set: $\{w(x,y) = 0\}$.

---

## 2. Mathematical Foundation

### 2.1 WEB-Spline Ansatz

**Weighted Extended B-Spline Approximation:**
$$u_h(x,y) = \sum_{i,j} c_{ij} \cdot w(x,y) \cdot B_i^n(x) B_j^n(y)$$

where:
- $B_i^n$ - B-spline basis function of degree $n$
- $c_{ij}$ - coefficient for B-spline centered at $(x_i, y_j)$
- $w(x,y)$ - weight function (ensures $u_h = 0$ on $\partial D$ automatically)

**Key Property:** Since $w(x,y) = 0$ on $\partial D$, the boundary condition $u = 0$ is **automatically satisfied** without additional constraints.

### 2.2 B-Spline Basis Functions

**Cardinal B-Splines:**
$$B^n(t) = \begin{cases}
B^0(t) = \begin{cases} 1 & 0 \leq t < 1 \\ 0 & \text{otherwise} \end{cases} \\
B^n(t) = \frac{t}{n} B^{n-1}(t) + \frac{n+1-t}{n} B^{n-1}(t-1)
\end{cases}$$

**Properties:**
- Support: $[0, n+1]$
- Smoothness: $C^{n-1}$ continuous
- Partition of unity: $\sum_i B_i^n(x) = 1$
- Tensor product: $\Phi_{ij}(x,y) = B_i^n(x) B_j^n(y)$

**Derivatives:**
$$\frac{d}{dt} B^n(t) = B^{n-1}(t) - B^{n-1}(t-1)$$
$$\frac{d^2}{dt^2} B^n(t) = B^{n-2}(t) - 2B^{n-2}(t-1) + B^{n-2}(t-2)$$

### 2.3 Lagrange Extension

To handle B-splines whose centers lie **outside** the domain, we use **Lagrange polynomial extension**.

**1D Extension Coefficient:**

For inner points $I = \{\alpha, \alpha+1, \ldots, \alpha+n\}$ and outer point $j$:
$$e_{i,j} = \prod_{\substack{k \in I \\ k \neq i}} \frac{j - k}{i - k}$$

This is the Lagrange basis polynomial $L_i(j)$ evaluated at $j$.

**2D Extension (Tensor Product):**
$$e_{(i_1,i_2), (j_1,j_2)} = e_{i_1,j_1} \cdot e_{i_2,j_2}$$

**Extension Formula:**

Coefficients for outer B-splines are determined by:
$$c_{\text{outer}} = \sum_{\text{inner } (n+1)\times(n+1) \text{ array}} e_{ij} \cdot c_{\text{inner}}$$

### 2.4 Collocation Equation

**Laplacian of Weighted B-Spline:**
$$-\Delta(w \cdot \Phi) = -w \Delta \Phi - 2\nabla w \cdot \nabla \Phi - \Phi \Delta w$$

Expanding:
$$-\Delta(w B_i B_j) = -w(B_i'' B_j + B_i B_j'') - 2(w_x B_i' B_j + w_y B_i B_j') - B_i B_j (w_{xx} + w_{yy})$$

**Collocation at Grid Points:**

At collocation point $(x_k, y_\ell)$:
$$\sum_{i,j} c_{ij} \left[-\Delta(w B_i B_j)\right]_{(x_k, y_\ell)} = f(x_k, y_\ell)$$

This gives a linear system: $\mathbf{A} \mathbf{c} = \mathbf{f}$.

---

## 3. Algorithm Overview

### 3.1 Main Steps

```
1. Classification (Identify Inner/Outer B-splines)
   └─ Evaluate weight function w on extended grid
   └─ Inner: w(center) > 0
   └─ Outer: w(center) ≤ 0 but support overlaps D
   └─ Identify (n+1)×(n+1) fully inner element arrays

2. Extension Matrix Construction
   └─ For each outer B-spline:
      └─ Find nearest inner element array
      └─ Compute Lagrange extension coefficients
      └─ Build extension matrix T: (m_inner × m_outer)

3. Collocation Matrix Assembly
   └─ For each B-spline pair (i,j):
      └─ Evaluate -Δ(w·B_i·B_j) at collocation points
      └─ Form banded array G[k1, k2, s1, s2]
   └─ Convert to sparse matrix SG
   └─ Apply extension: A_reduced = SG[I,I] + SG[I,J]·T^T

4. Solve Linear System
   └─ Solve A_reduced · c_inner = f_inner
   └─ Extend to outer: c_outer = T^T · c_inner

5. Evaluate Solution
   └─ u_h(x,y) = Σ c_ij · w(x,y) · B_i(x) · B_j(y)
```

### 3.2 Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Classification | $O((H+n)^2)$ | Evaluate w on grid |
| Extension Matrix | $O(m_j \cdot (n+1)^2)$ | $m_j$ = # outer B-splines |
| Matrix Assembly | $O(H^2 \cdot (2bw+1)^2)$ | bw = bandwidth |
| Solve | $O(m_i^3)$ or $O(m_i)$ | Dense: $O(m_i^3)$, Sparse iter: $O(m_i)$ |
| Evaluation | $O((H+n)^2 \cdot (2bw+1)^2)$ | Sum over basis |

**Total:** $O(H^2 n^2 + m_i^3)$ where $m_i \approx H^2$ for typical domains.

---

## 4. Core Components

### 4.1 B-Spline Evaluation

**Function:** `bspline_evaluate(t, n)`
- Evaluates $B^n(t)$ using Cox-de Boor recursion
- Input: evaluation points $t$, degree $n$
- Output: B-spline values

**Function:** `bspline_derivative(t, n, order)`
- Computes $\frac{d^k}{dt^k} B^n(t)$ for $k=1,2$
- Uses derivative formula: $\frac{d}{dt} B^n = B^{n-1}(t) - B^{n-1}(t-1)$

### 4.2 Extension Coefficient Computation

**Function:** `extension_coefficient_1d(I_indices, j)`
- Computes 1D Lagrange coefficients
- Input: inner indices $I$, outer index $j$
- Output: $e_{i,j}$ for each $i \in I$

**Function:** `compute_collocation_data(n, J_MAX)`
- Precomputes all extension coefficients
- Precomputes B-spline products: $b_{00}, b_{10}, b_{01}, b_{20}, b_{02}$
- Stores in dictionary `CD` for efficient reuse

**Output Structure:**
```python
CD = {
    'E': extension_coeffs,  # Shape: ((n+1)^2, dim, dim, 2, 2)
    'b00': B_i · B_j,
    'b10': B_i' · B_j,
    'b01': B_i · B_j',
    'b20': B_i'' · B_j,
    'b02': B_i · B_j'',
    'n': degree,
    'bw': bandwidth,
    'dim': max distance
}
```

### 4.3 Weight Functions

**Base Class:** `WeightFunction`
- Abstract interface for domain definition
- Must implement: `__call__(x, y)` → `(w, wx, wy, wxx, wyy)`

**Built-in Implementations:**

1. **DiscWeightFunction:**
   $$w = 1 - (2x-1)^2 - (2y-1)^2$$
   Domain: unit disc centered at $(0.5, 0.5)$

2. **ShovelWeightFunction:**
   $$w = 1 - p^2 - (q + p^2 - 1)^2$$
   where $p = 2x-1$, $q = 2.25y - 0.25$

3. **NeuralWeightFunction:**
   - Uses PyTorch neural network for SDF
   - Computes derivatives via autograd
   - Supports coordinate transformations
   - Supports SDF output transforms (sigmoid, tanh, etc.)

**Neural Weight Function Features:**
- **Domain Mapping:** Maps $[0,1]^2 \to [x_1, x_2] \times [y_1, y_2]$
- **Autodiff:** Automatic gradient computation via PyTorch
- **Transforms:** Optional smoothing transforms on SDF output
- **Chain Rule:** Proper derivative scaling for coordinate transforms

### 4.4 Array to Matrix Conversion

**Function:** `array_to_matrix(G)`
- Converts banded 4D array to sparse matrix
- Input: `G[k1, k2, s1, s2]` - matrix entry at row $(k_1, k_2)$ and column offset $(s_1-bw, s_2-bw)$
- Output: CSR sparse matrix
- Handles boundary conditions by setting invalid indices to -1

**Indexing Convention:**
- Python uses row-major (C-order): $\text{linear\_idx} = k_1 \cdot \text{dim} + k_2$
- MATLAB uses column-major (Fortran-order): $\text{linear\_idx} = k_1 + k_2 \cdot \text{dim}$
- Code carefully adapted for Python conventions

---

## 5. Implementation Details

### 5.1 Main Solver Function

**Function:** `collocation_2d(n, H, wfct, f, CD, verbose, domain)`

**Parameters:**
- `n`: B-spline degree (2-5)
- `H`: grid resolution (grid width = $1/H$)
- `wfct`: weight function object
- `f`: right-hand side function
- `CD`: precomputed collocation data (optional)
- `verbose`: print progress
- `domain`: physical domain bounds (optional)

**Returns:**
- `Uxy`: solution values on grid
- `xB, yB`: grid coordinates
- `con`: condition number
- `dim_sys`: system dimension
- `rtimes`: timing breakdown

### 5.2 Grid and Indexing

**Extended Grid:**
```python
t_range = [-(n-1)/2 - bw, ..., H + (n-1)/2 + bw] / H
```
- Includes ghost points for extension
- Total size: $(H + n + 2bw + 1)$ points per direction

**Collocation Point Indices:**
```python
indB = [bw, bw+1, ..., H+n+bw-1]  # Python 0-based
```
- These are the $H+n$ B-spline centers

**Inner B-splines:**
- Centers where $w(x_i, y_j) > 0$
- Indices stored in `IL` (linear indices)
- Count: $m_i$

**Outer B-splines:**
- Centers where $w \leq 0$ but support overlaps domain
- Indices stored in `JL`
- Count: $m_j$

### 5.3 Classification Details

**Inner Element Arrays:**
- $(n+1) \times (n+1)$ blocks of B-splines all with centers in $D$
- Used as reference points for Lagrange extension
- Centers: $(xe, ye) = ((k+0.5)/H, (\ell+0.5)/H)$ for $k,\ell = 0, \ldots, H-1$

**Extension Logic:**

For each outer B-spline at $(j_1, j_2)$:
1. Find nearest inner element array center $(xe, ye)$
2. Compute distance $D = [j_1, j_2] - \text{array\_corner} - n/2$
3. Get sign and magnitude: $D_0 = \text{sign}(D)$, $D_1 = |\lfloor D \rfloor|$
4. Extract extension coefficients: `E[:, D1[0], D1[1], D0[0], D0[1]]`
5. Map to inner B-splines in array and populate extension matrix $T$

### 5.4 Matrix Assembly

**Collocation Matrix Entries:**

For B-spline pair $(i,j)$ and collocation point $(k,\ell)$:
$$G[k,\ell; i,j] = \left[-\Delta(w B_i B_j)\right]_{(x_k, y_\ell)}$$

**Expanded Form:**
$$G = -H^2(b_{20} + b_{02}) w - b_{00}(w_{xx} + w_{yy}) - 2H(b_{10} w_x + b_{01} w_y)$$

where:
- $b_{20} = B_i'' B_j$, $b_{02} = B_i B_j''$ (second derivatives)
- $b_{10} = B_i' B_j$, $b_{01} = B_i B_j'$ (first derivatives)
- $b_{00} = B_i B_j$ (values)
- All evaluated at offset $(s_1, s_2)$ from collocation point

**Storage:**

Banded array `G[k1, k2, s1, s2]`:
- $(k_1, k_2)$: collocation point index
- $(s_1, s_2)$: offset to B-spline center ($s_1, s_2 \in [-bw, bw]$)
- $bw = \lfloor n/2 \rfloor$ (bandwidth)

### 5.5 Domain Transformation

**Coordinate Mapping:**

From computational domain $[0,1]^2$ to physical domain $[x_1, x_2] \times [y_1, y_2]$:
$$x_{\text{phys}} = x_1 + x_{\text{grid}} \cdot (x_2 - x_1)$$
$$y_{\text{phys}} = y_1 + y_{\text{grid}} \cdot (y_2 - y_1)$$

**Laplacian Scaling:**

The Laplacian transforms as:
$$\Delta_{\text{grid}} = s_x^2 \frac{\partial^2}{\partial x_{\text{phys}}^2} + s_y^2 \frac{\partial^2}{\partial y_{\text{phys}}^2}$$

For isotropic domains ($s_x = s_y = s$):
$$-\Delta_{\text{phys}} u = f \quad \Rightarrow \quad -s^2 \Delta_{\text{phys}} u = s^2 f$$
$$\Rightarrow \quad -\Delta_{\text{grid}} u = s^2 \cdot f$$

**Note:** The code uses `laplacian_scale = scale_x * scale_y`, which equals $s^2$ for isotropic domains.

**Implementation:**
```python
laplacian_scale = scale_x * scale_y  # = (x2-x1) * (y2-y1)
f_transformed = lambda x, y: laplacian_scale * f_original(x_phys, y_phys)
```

### 5.6 Neural SDF Transforms

**Available Transforms:**

1. **Sigmoid:** $g(w) = \frac{1}{1+e^{-w}}$
   - $g'(w) = g(w)(1-g(w))$
   - $g''(w) = g'(w)(1-2g(w))$

2. **Tanh:** $g(w) = \tanh(w)$
   - $g'(w) = 1 - \tanh^2(w)$
   - $g''(w) = -2\tanh(w) \cdot \text{sech}^2(w)$

3. **Logarithmic:** $g(w) = \log(w+1)$
   - $g'(w) = 1/(w+1)$
   - $g''(w) = -1/(w+1)^2$

4. **Trapezoid:** $g(w) = \min(w \cdot \text{tang}, 1)$
   - $g'(w) = \text{tang}$ if $w \cdot \text{tang} < 1$ else $0$
   - $g''(w) = 0$

**Chain Rule Application:**

For $w_{\text{new}} = g(w)$:
$$\frac{\partial w_{\text{new}}}{\partial x} = g'(w) \frac{\partial w}{\partial x}$$
$$\frac{\partial^2 w_{\text{new}}}{\partial x^2} = g''(w) \left(\frac{\partial w}{\partial x}\right)^2 + g'(w) \frac{\partial^2 w}{\partial x^2}$$

---

## 6. Usage Guide

### 6.1 Basic Example

```python
from collocation_WEB import *

# Define weight function (unit disc)
wfct = DiscWeightFunction()

# Define load function (for case 0: exp(w)-1)
def f_rhs(x, y):
    w, wx, wy, wxx, wyy = wfct(x, y)
    return -np.exp(w) * (wx**2 + wy**2 + wxx + wyy)

# Solve
n = 3  # degree
H = 40  # resolution
Uxy, xB, yB, con, dim, rtimes = collocation_2d(n, H, wfct, f_rhs)

print(f"Solution computed: {Uxy.shape}")
print(f"Condition number: {con:.2e}")
```

### 6.2 With Neural SDF

```python
import torch
from Geomertry import AnaliticalDistanceCircle

# Load neural/analytical model
model = AnaliticalDistanceCircle()

# Create weight function with domain transform
wfct = NeuralWeightFunction(
    model=model,
    domain={'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1},
    transform='tanh'  # Optional smoothing
)

# Define load in physical coordinates
def f_physical(x, y):
    return -8 * x  # Example load

# Solve (domain transform handled automatically)
Uxy, xB, yB, con, dim, rtimes = collocation_2d(
    n=3, H=40, wfct=wfct, f=f_physical,
    domain={'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1}
)
```

### 6.3 Convergence Study

```python
# Run convergence study
results = convergence_study(
    n_list=[2, 3, 4],
    H_list=[10, 20, 40, 80],
    domain='disc',
    function_case=0
)

# Results stored in dict: results[(n, H)] = {...}
# Contains: ErrMax, ErrL2, MAE, H1_semi, H1_error, condition, rtimes
```

### 6.4 Error Metrics

**Computed Errors:**

1. **L∞ Error (Relative):**
   $$\text{ErrMax} = \frac{\max |u_h - u_{\text{exact}}|}{\max |u_{\text{exact}}|}$$

2. **L² Error (Relative):**
   $$\text{ErrL2} = \frac{\|u_h - u_{\text{exact}}\|_{L^2}}{\|u_{\text{exact}}\|_{L^2}}$$

3. **Mean Absolute Error:**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |u_h(x_i) - u_{\text{exact}}(x_i)|$$

4. **H¹ Semi-norm Error:**
   $$\text{H1\_semi} = \frac{\|\nabla(u_h - u_{\text{exact}})\|_{L^2}}{\|\nabla u_{\text{exact}}\|_{L^2}}$$

5. **Full H¹ Error:**
   $$\text{H1\_error} = \sqrt{\text{ErrL2}^2 + \text{H1\_semi}^2}$$

---

## 7. References

### 7.1 Primary References

1. **Höllig, K., Reif, U., Wipper, J. (2001)**
   "Weighted Extended B-Spline Approximation of Dirichlet Problems"
   *SIAM Journal on Numerical Analysis*, 39(2), 442-462.

2. **Apprich, C., Höllig, K., Hörner, J., Reif, U.**
   "Collocation with WEB-Splines" (CwBS-Programs Version 1.0)
   MATLAB implementation and technical documentation.

### 7.2 Related Work

3. **De Boor, C. (2001)**
   "A Practical Guide to Splines"
   *Springer-Verlag*, Revised Edition.
   (B-spline theory and algorithms)

4. **Höllig, K. (2003)**
   "Finite Element Methods with B-Splines"
   *SIAM Frontiers in Applied Mathematics*.

5. **Bazilevs, Y., Calo, V.M., Cottrell, J.A., et al. (2010)**
   "Isogeometric Analysis: Approximation, Stability and Error Estimates"
   *Mathematical Models and Methods in Applied Sciences*, 20(12), 2199-2219.
   (IGA context and B-spline FEM)

### 7.3 Neural Implicit Representations

6. **Sitzmann, V., et al. (2020)**
   "Implicit Neural Representations with Periodic Activation Functions"
   *NeurIPS 2020* (SIREN networks for SDFs)

7. **Park, J.J., et al. (2019)**
   "DeepSDF: Learning Continuous Signed Distance Functions"
   *CVPR 2019* (Neural SDFs)

---

## Appendix A: Algorithm Pseudocode

```
Algorithm: WEB-Spline Collocation

Input:
  - n: B-spline degree
  - H: grid resolution
  - w(x,y): weight function (+ derivatives)
  - f(x,y): load function

Output:
  - u_h(x,y): approximate solution

1. Precompute Collocation Data:
   E[...] ← extension_coefficients(n)
   b00, b10, b01, b20, b02 ← bspline_products(n)

2. Create Extended Grid:
   t ← [-(n-1)/2-bw : H+(n-1)/2+bw] / H
   (x_grid, y_grid) ← meshgrid(t, t)
   
3. Evaluate Weight Function:
   (w, wx, wy, wxx, wyy) ← w(x_grid, y_grid)

4. Classification:
   IL ← {(i,j) : w(x_i, y_j) > 0}  # Inner B-splines
   JL ← {(i,j) : w(x_i, y_j) ≤ 0 AND support overlaps D}  # Outer
   eI ← {element arrays fully inside D}

5. Build Extension Matrix T (m_inner × m_outer):
   for each outer B-spline (j1, j2) in JL:
      Find nearest inner array center (xe, ye)
      Compute distance D = [(j1,j2) - array_corner - n/2]
      Get E_coeffs from precomputed E[..., D1, D0]
      Map to inner B-splines and populate T

6. Assemble Collocation Matrix G:
   for s1 in [-bw, ..., bw]:
      for s2 in [-bw, ..., bw]:
         G[:,:,s1+bw,s2+bw] = -Δ(w·B_s1·B_s2) at collocation points
         = -H²(b20+b02)·w - b00·(wxx+wyy) - 2H(b10·wx + b01·wy)

7. Convert to Sparse Matrix:
   SG ← array_to_matrix(G)  # (dim_B² × dim_B²)

8. Apply Extension:
   A_inner ← SG[IL,IL] + SG[IL,JL] · T^T

9. Solve System:
   c_inner ← solve(A_inner, f[IL])
   c_outer ← T^T · c_inner

10. Evaluate Solution:
    for all grid points (x,y):
       u_h(x,y) = Σ c_ij · w(x,y) · B_i(x) · B_j(y)

Return u_h
```

---

## Appendix B: Key Implementation Functions

| Function | Purpose | Complexity |
|----------|---------|------------|
| `bspline_evaluate(t, n)` | Evaluate $B^n(t)$ | $O(n \cdot |t|)$ |
| `bspline_derivative(t, n, order)` | Compute $\frac{d^k}{dt^k} B^n(t)$ | $O(n \cdot |t|)$ |
| `extension_coefficient_1d(I, j)` | Lagrange coefficients | $O(n^2)$ |
| `compute_collocation_data(n)` | Precompute E, b | $O(n^4)$ (one-time) |
| `array_to_matrix(G)` | Banded array → sparse | $O(H^2 \cdot bw^2)$ |
| `collocation_2d(...)` | Main solver | $O(H^2 n^2 + m_i^3)$ |

---

## Appendix C: Test Problem Cases

| Case | Solution | Domain | Notes |
|------|----------|--------|-------|
| 0 | $u = e^w - 1$ | Any | Default WEB-spline test |
| 1 | $u = x(x^2+y^2-1)$ | Circle | Simple polynomial |
| 2 | $u = \cos(\frac{\pi}{2}(x^2+y^2))$ | Circle | Trigonometric |
| 3 | $u = x(x^2+y^2-1) + 2$ | Circle | With offset |
| 4 | $u = x(x^2+y^2-1) + x + 2y$ | Circle | With linear |
| 5 | $u = \sin(2\pi x)\sin(2\pi y)$ | L-shape | Benchmark |
| 7 | $u = \sin(\pi(x^2+y^2))$ | Double circle | Complex |

---

**Document Version:** 1.0  
**Date:** January 27, 2026  
**Implementation:** Python 3.9+, NumPy, SciPy, PyTorch  
**File:** `collocation_WEB.py` (~1474 lines)
