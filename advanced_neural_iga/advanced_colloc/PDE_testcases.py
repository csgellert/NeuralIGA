import numpy as np
import torch
import math
import mesh
from typing import Optional, Tuple
# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)
NP_DTYPE = np.float64
TORCH_DTYPE = torch.float64


FUNCTION_CASE = 11

# Active geometry model used by manufactured testcases (set inside GaussQuadrature).
ACTIVE_MODEL: Optional[torch.nn.Module] = None

# =============================================================================
# Pentagon boundary geometry for FUNCTION_CASE 11/12 (5-sided domains)
# =============================================================================
# radius/center/rotation place the 5 corners; for sides listed in
# curved_side_indices the span bulges inward as a cubic Bezier-equivalent
# curve (see inhomogenous_boundary._build_polygon_boundary_data), all other
# sides stay straight. Case 11 is an all-straight pentagon inscribed in the
# unit circle; case 12 is the same pentagon with 3 curved and 2 straight sides.
# IMPORTANT: whatever per-side distance model is passed to get_lifting /
# get_lifting_polygon (inhomogenous_boundary.py) must be built with this same
# rotation, otherwise its side order will not line up with the Dirichlet data
# prescribed below and side i's distance will be blended against the wrong
# side's boundary value.
#
# side_placement chooses how inhomogenous_boundary picks, for a given query
# point, the representative point on side i at which the Dirichlet formula is
# evaluated:
# - "distance_ratio": case 11's original scheme. Placed by a ratio of the
#   model's own raw side distances (see inhomogenous_boundary._ratio_side_param),
#   not an actual nearest-point projection. Kept exactly as originally
#   validated (e.g. against the case-11 CSV ground truth) -- do not change
#   this to "nearest_point" without re-validating, it measurably shifts
#   lifting values away from that ground truth.
# - "nearest_point": case 12's original (and more geometrically standard)
#   scheme, projecting onto a sampled polyline of the side curve.
CASE_POLYGON_GEOMETRY = {
    11: {
        "radius": 1.0, "center": (0.0, 0.0), "rotation": 0.0, "bulge": 0.0,
        "samples_per_side": 128, "curved_side_indices": (), "side_placement": "distance_ratio",
    },
    12: {
        "radius": 0.5, "center": (0.0, 0.0), "rotation": 0.0, "bulge": 0.18,
        "samples_per_side": 128, "curved_side_indices": (0, 2, 4), "side_placement": "nearest_point",
    },
}

# Sides forced to zero Dirichlet data, keyed by FUNCTION_CASE. This is the
# single place that assigns per-side boundary values for the pentagon cases;
# edit it (and dirichletBoundary_side_vectorized below) to reassign boundary
# data without touching inhomogenous_boundary.py.
CASE_SIDE_ZERO_DIRICHLET = {
    11: (0, 4),
    12: (1, 3),
}


def get_case_polygon_geometry(function_case: int) -> dict:
    """Pentagon side geometry (radius/center/rotation/bulge/curved sides)."""
    if function_case not in CASE_POLYGON_GEOMETRY:
        raise NotImplementedError(f"No polygon geometry defined for FUNCTION_CASE={function_case}.")
    return CASE_POLYGON_GEOMETRY[function_case]


def _transform_scalar_and_derivs(
    d_raw: torch.Tensor, transform: Optional[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (w, T'(d), T''(d)) for supported transforms.

    This mirrors mesh.distance_with_derivative_vect_trasformed for the *value*.
    We additionally provide second derivatives wrt the scalar argument so we can
    compute w_xx, w_yy via chain rule.
    """
    if transform is None:
        w = d_raw
        tp = torch.ones_like(d_raw)
        tpp = torch.zeros_like(d_raw)
        return w, tp, tpp

    t = str(transform).lower()
    if t == "sigmoid":
        w = 1.0 / (1.0 + torch.exp(-d_raw))
        tp = w * (1.0 - w)
        tpp = tp * (1.0 - 2.0 * w)
        return w, tp, tpp
    if t == "tanh":
        w = torch.tanh(d_raw)
        tp = 1.0 - w * w
        tpp = -2.0 * w * tp
        return w, tp, tpp
    if t == "logarithmic":
        w = torch.log(d_raw + 1.0)
        tp = 1.0 / (d_raw + 1.0)
        tpp = -1.0 / (d_raw + 1.0) ** 2
        return w, tp, tpp
    if t == "exponential":
        expd = torch.exp(d_raw)
        w = expd - 1.0
        tp = expd
        tpp = expd
        return w, tp, tpp

    raise NotImplementedError(
        f"Second-derivative manufactured cases do not support transform={transform!r}. "
        "Use TRANSFORM=None or one of: None, 'sigmoid', 'tanh', 'logarithmic', 'exponential'."
    )


def _eval_weight_and_hessian_diag_np(
    x: np.ndarray,
    y: np.ndarray,
    model: torch.nn.Module,
    transform: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate transformed weight w and derivatives (wx, wy, wxx, wyy) at points.

    Uses autograd on the given ``model``.
    """
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=NP_DTYPE), np.asarray(y, dtype=NP_DTYPE))
    out_shape = x_arr.shape
    x_np = x_arr.reshape(-1)
    y_np = y_arr.reshape(-1)
    crd = torch.tensor(np.stack([x_np, y_np], axis=1), dtype=TORCH_DTYPE, requires_grad=True)

    d_raw = model(crd).reshape(-1)
    grad = torch.autograd.grad(
        outputs=d_raw,
        inputs=crd,
        grad_outputs=torch.ones_like(d_raw),
        create_graph=True,
        retain_graph=True,
    )[0]
    dx_raw = grad[:, 0]
    dy_raw = grad[:, 1]

    dxx_raw = torch.autograd.grad(
        outputs=dx_raw,
        inputs=crd,
        grad_outputs=torch.ones_like(dx_raw),
        create_graph=False,
        retain_graph=True,
    )[0][:, 0]
    dyy_raw = torch.autograd.grad(
        outputs=dy_raw,
        inputs=crd,
        grad_outputs=torch.ones_like(dy_raw),
        create_graph=False,
        retain_graph=False,
    )[0][:, 1]

    w, tp, tpp = _transform_scalar_and_derivs(d_raw, transform)

    wx = tp * dx_raw
    wy = tp * dy_raw
    wxx = tpp * (dx_raw ** 2) + tp * dxx_raw
    wyy = tpp * (dy_raw ** 2) + tp * dyy_raw

    w_np = w.detach().cpu().numpy().reshape(out_shape)
    wx_np = wx.detach().cpu().numpy().reshape(out_shape)
    wy_np = wy.detach().cpu().numpy().reshape(out_shape)
    wxx_np = wxx.detach().cpu().numpy().reshape(out_shape)
    wyy_np = wyy.detach().cpu().numpy().reshape(out_shape)
    return (w_np, wx_np, wy_np, wxx_np, wyy_np)



def get_domain_for_case(function_case: int) -> dict:
    """Return the physical domain box for a given FUNCTION_CASE.

    Notes:
        - Case 0 matches the original WEB-spline disc example defined on [0,1]^2.
        - Cases 1..7 use the [-1,1]^2 box.
        - Case 8 uses the [-4,4]x[-3,3] box.
        - Cases 9..10 use the [-1,1]^2 box (triangle, pentagon).
    """
    if function_case == 0:
        return {"x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0}
    if 1 <= function_case <= 7:
        return {"x1": -1.0, "x2": 1.0, "y1": -1.0, "y2": 1.0}
    if 9 <= function_case <= 10:
        return {"x1": -1.0, "x2": 1.0, "y1": -1.0, "y2": 1.0}
    if function_case == 8:
        return {"x1": -4.0, "x2": 4.0, "y1": -3.0, "y2": 3.0}
    if function_case in (11, 12):
        return {"x1": -1, "x2": 1, "y1": -1, "y2": 1}
    raise NotImplementedError(f"Unknown FUNCTION_CASE={function_case}")


def set_function_case(function_case: int) -> None:
    """Set FUNCTION_CASE and keep DOMAIN consistent."""
    global FUNCTION_CASE, DOMAIN
    FUNCTION_CASE = int(function_case)
    DOMAIN = get_domain_for_case(FUNCTION_CASE)


DOMAIN = get_domain_for_case(FUNCTION_CASE)

# Vectorized function evaluations
def load_function_vectorized(x, y):
    """Vectorized version of load_function"""
    if FUNCTION_CASE == 0:
        # WEB-spline disc example on [0,1]^2:
        # u = exp(w) - 1 with w = 1 - (2x-1)^2 - (2y-1)^2
        # f = -Δu = -exp(w) * (|∇w|^2 + Δw)
        wx = -4 * (2 * x - 1)
        wy = -4 * (2 * y - 1)
        wxx = -8.0
        wyy = -8.0
        w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
        return -np.exp(w) * (wx ** 2 + wy ** 2 + wxx + wyy)
    if FUNCTION_CASE == 1:
        return -8*x
    elif FUNCTION_CASE == 2:
        arg = (x**2 + y**2)*math.pi/2
        return -(-2*math.pi*np.sin(arg)-np.cos(arg)*(x**2 + y**2)*math.pi**2)
    elif FUNCTION_CASE == 3:
        return -8*x
    elif FUNCTION_CASE == 4:
        return -8*x
    elif FUNCTION_CASE == 5:#L-shape
        return 8*math.pi*math.pi*np.sin(2*math.pi*x)*np.sin(2*math.pi*y)
    elif FUNCTION_CASE == 6: #tube
        return -(x**2 + y**2)
    elif FUNCTION_CASE == 7: #double circle
        arg = (x**2 + y**2)*math.pi
        return 4*math.pi*np.cos(arg) -4*math.pi*arg*np.sin(arg)
    elif FUNCTION_CASE == 8: # http://www.web-spline.de/examples/analytic/index.html
        e = 1-(x**2)/16 -(y**2)/9
        ex = -x/8
        exx = -1/8
        ey = -2*y/9
        eyy = -2/9
        k = x**2+1.5*x+y**2-y-3/16
        kx = 2*x +1.5
        kxx = 2
        ky = 2*y -1
        kyy = 2

        tmp1 = -np.sin(e*k*0.5)*0.25*((ex*k + e*kx)**2 + (ey*k + e*ky)**2)
        tmp2 = np.cos(e*k*0.5)*0.5*( (exx*k + 2*ex*kx + e*kxx) + (eyy*k + 2*ey*ky + e*kyy) )
        return -(tmp1 + tmp2)
    elif FUNCTION_CASE in (9, 10):
        # Manufactured Poisson on polygon domains with homogeneous Dirichlet on w=0.
        # We pick u_ex = w^2 so u_ex|_{\partial\Omega}=0 by construction.
        # f = -Δu = -[2(|∇w|^2) + 2w(Δw)].
        if ACTIVE_MODEL is None:
            raise RuntimeError(
                "ACTIVE_MODEL is not set. Call the FEM assembly routines with a model so GaussQuadrature can set ACTIVE_MODEL."
            )
        w, wx, wy, wxx, wyy = _eval_weight_and_hessian_diag_np(x, y, ACTIVE_MODEL, transform=mesh.TRANSFORM)
        return -(2.0 * (wx ** 2 + wy ** 2) + 2.0 * w * (wxx + wyy))
    elif FUNCTION_CASE == 11:
        return np.sin(y * np.pi)
    elif FUNCTION_CASE == 12:
        return np.sin(y * np.pi)
    else:
        raise NotImplementedError

def dirichletBoundary_vectorized(x, y):
    """Vectorized version of dirichletBoundary"""
    if FUNCTION_CASE == 0:
        return np.zeros_like(x)
    if FUNCTION_CASE == 1:
        return np.full_like(x, 0)
    if FUNCTION_CASE == 2:
        return np.zeros_like(x)
    if FUNCTION_CASE == 3:
        return np.full_like(x, 2)
    if FUNCTION_CASE == 4:
        return x + 2*y
    elif FUNCTION_CASE == 5:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 7:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 8:
        return np.zeros_like(x)
    elif FUNCTION_CASE in (9, 10):
        return np.zeros_like(x)
    elif FUNCTION_CASE == 11:
        return 1**2 - (x**2 + y**2)
    elif FUNCTION_CASE == 12:
        return 1**2 - (x**2 + y**2)
    else:
        raise NotImplementedError

def dirichletBoundary_side_vectorized(x, y, side_idx):
    """Per-side Dirichlet data for the 5-sided FUNCTION_CASE 11/12 domains.

    Starts from ``dirichletBoundary_vectorized`` and forces the sides listed
    in ``CASE_SIDE_ZERO_DIRICHLET[FUNCTION_CASE]`` to zero.

    Accepts either numpy arrays or torch tensors for x/y (cases 11/12's
    Dirichlet formula is plain arithmetic, so dirichletBoundary_vectorized
    works unchanged either way). Torch input keeps its autograd graph, which
    inhomogenous_boundary.py's "distance_ratio" side placement (case 11)
    relies on for the boundary lifting function's exact Laplacian.
    """
    vals = dirichletBoundary_vectorized(x, y)
    zero_sides = CASE_SIDE_ZERO_DIRICHLET.get(FUNCTION_CASE)
    if not zero_sides:
        return vals
    if torch.is_tensor(vals):
        side_idx_t = side_idx if torch.is_tensor(side_idx) else torch.as_tensor(side_idx, device=vals.device)
        zero_mask = torch.zeros_like(side_idx_t, dtype=torch.bool)
        for side in zero_sides:
            zero_mask = zero_mask | (side_idx_t == side)
        return torch.where(zero_mask, torch.zeros_like(vals), vals)
    zero_mask = np.isin(np.asarray(side_idx), zero_sides)
    return np.where(zero_mask, 0.0, vals)

def dirichletBoundaryDerivativeX_vectorized(x, y):
    """Vectorized version of dirichletBoundaryDerivativeX"""
    if FUNCTION_CASE <= 3:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 4:
        return np.ones_like(x)
    elif FUNCTION_CASE == 5:#L-shape
        return np.zeros_like(x)
    elif FUNCTION_CASE == 7:
        return np.zeros_like(x)
    elif FUNCTION_CASE ==8:
        return np.zeros_like(x)
    elif FUNCTION_CASE in (9, 10):
        return np.zeros_like(x)
    elif FUNCTION_CASE == 12:
        return -2.0 * x
    else: 
        raise NotImplementedError

def dirichletBoundaryDerivativeY_vectorized(x, y):
    """Vectorized version of dirichletBoundaryDerivativeY"""
    if FUNCTION_CASE <= 3:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 4:
        return np.full_like(x, 2)
    elif FUNCTION_CASE ==5:  #L-shape
        return np.zeros_like(x)
    elif FUNCTION_CASE == 7:
        return np.zeros_like(x)
    elif FUNCTION_CASE == 8:
        return np.zeros_like(x)
    elif FUNCTION_CASE in (9, 10):
        return np.zeros_like(x)
    elif FUNCTION_CASE == 12:
        return -2.0 * y
    else: 
        raise NotImplementedError

def dirichletBoundaryDerivativeXX_vectorized(x, y):
        """Vectorized second derivative d²g/dx² of the prescribed Dirichlet data g(x,y).

        Notes:
        - For most existing FUNCTION_CASE values the Dirichlet data is either zero,
            constant, or linear, hence the second derivatives are identically zero.
        - This function is defined over the whole computational domain as an
            extension of boundary data, which is required by the collocation_WEB
            non-homogeneous Dirichlet formulation.
        """
        # All currently supported Dirichlet boundary functions are at most linear.
        return np.zeros_like(x)

def dirichletBoundaryDerivativeYY_vectorized(x, y):
        """Vectorized second derivative d²g/dy² of the prescribed Dirichlet data g(x,y).

        See dirichletBoundaryDerivativeXX_vectorized for notes.
        """
        # All currently supported Dirichlet boundary functions are at most linear.
        return np.zeros_like(x)



def solution_function(x,y):
    if FUNCTION_CASE == 0:
        w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
        return np.exp(w) - 1
    if FUNCTION_CASE == 1:
        return x*(x**2 + y**2 -1)
    elif FUNCTION_CASE == 2:
        return np.cos((x**2 + y**2)*np.pi/2)
    elif FUNCTION_CASE == 3:
        return x*(x**2 + y**2 -1) + 2
    elif FUNCTION_CASE == 4:
        return x*(x**2 + y**2 -1) + x +2*y
    elif FUNCTION_CASE == 5: #L-shape
        return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    elif FUNCTION_CASE == 7: #double circle
        return math.sin(math.pi*(x**2 + y**2))
    elif FUNCTION_CASE ==8:
        e = 1-(x**2)/16 -(y**2)/9
        k = x**2+1.5*x+y**2-y-3/16
        return np.sin(e*k*0.5)
    elif FUNCTION_CASE in (9, 10):
        if ACTIVE_MODEL is None:
            raise RuntimeError("ACTIVE_MODEL is not set for FUNCTION_CASE 9/10.")
        w, *_ = _eval_weight_and_hessian_diag_np(x, y, ACTIVE_MODEL, transform=mesh.TRANSFORM)
        return w ** 2
    elif FUNCTION_CASE in (11, 12):
        return np.zeros_like(x)  # Placeholder for the solution function in case 11
    else: raise NotImplementedError
def solution_function_derivative_x(x,y):
    if FUNCTION_CASE == 0:
        w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
        wx = -4 * (2 * x - 1)
        return np.exp(w) * wx
    if FUNCTION_CASE == 1:
        return 3*x**2 + y**2 -1
    elif FUNCTION_CASE == 2:
        arg = (x**2 + y**2)*np.pi/2
        return -np.pi*x*np.sin(arg)
    elif FUNCTION_CASE == 3:
        return 3*x**2 + y**2 -1
    elif FUNCTION_CASE == 4:
        return 3*x**2 + y**2 -1 +1
    elif FUNCTION_CASE == 5: #L-shape
        return 2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    elif FUNCTION_CASE ==7: #double circle
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi
        return 2*math.pi*x*math.cos(arg)
    elif FUNCTION_CASE ==8:
        e = 1-(x**2)/16 -(y**2)/9
        ex = -x/8
        k = x**2+1.5*x+y**2-y-3/16
        kx = 2*x +1.5
        return 0.5*np.cos(e*k*0.5)*(ex*k + e*kx)
    elif FUNCTION_CASE in (9, 10):
        if ACTIVE_MODEL is None:
            raise RuntimeError("ACTIVE_MODEL is not set for FUNCTION_CASE 9/10.")
        w, wx, *_ = _eval_weight_and_hessian_diag_np(x, y, ACTIVE_MODEL, transform=mesh.TRANSFORM)
        return 2.0 * w * wx
    else: raise NotImplementedError
def solution_function_derivative_y(x,y):
    if FUNCTION_CASE == 0:
        w = 1 - (2 * x - 1) ** 2 - (2 * y - 1) ** 2
        wy = -4 * (2 * y - 1)
        return np.exp(w) * wy
    if FUNCTION_CASE == 1:
        return 2*x*y
    elif FUNCTION_CASE == 2:
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi/2
        return -math.pi*y*math.sin(arg)
    elif FUNCTION_CASE == 3:
        return 2*x*y
    elif FUNCTION_CASE == 4:
        return 2*x*y +2
    elif FUNCTION_CASE == 5: #L-shape
        return 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    elif FUNCTION_CASE ==7: #double circle
        raise NotImplementedError
        arg = (x**2 + y**2)*math.pi
        return 2*math.pi*y*math.cos(arg)
    elif FUNCTION_CASE ==8:
        e = 1-(x**2)/16 -(y**2)/9
        ey = -2*y/9
        k = x**2+1.5*x+y**2-y-3/16
        ky = 2*y -1
        return 0.5*np.cos(e*k*0.5)*(ey*k + e*ky)
    elif FUNCTION_CASE in (9, 10):
        if ACTIVE_MODEL is None:
            raise RuntimeError("ACTIVE_MODEL is not set for FUNCTION_CASE 9/10.")
        w, _, wy, *_ = _eval_weight_and_hessian_diag_np(x, y, ACTIVE_MODEL, transform=mesh.TRANSFORM)
        return 2.0 * w * wy
    else: raise NotImplementedError
