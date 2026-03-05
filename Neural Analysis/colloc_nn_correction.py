"""Neural residual correction utilities for WEB-spline collocation results.

This module is intentionally notebook-agnostic: the notebook should only call
high-level functions from here.

We train a correction of the form:
  u_corr(x,y) = u_base(x,y) + w_+(x,y) * exp(-alpha*w_+(x,y)) * phi(features(x,y))

Features for convex polygons (triangle/pentagon):
  - per-edge distance to segment
  - per-edge sweep parameter t in [0,1] (projection along the closest point)

Residual (physical coordinates):
  r = -(Δ u_base + Δ corr) - f
where:
  - Δ u_base is approximated by finite differences on the collocation grid
  - Δ corr is computed by autograd (requires the geometry weight model to be torch)

Designed for manufactured problems where u_exact is known for selection/guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Local module in "Neural Analysis" folder.
import network_defs


ArrayFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def set_seed(seed: int = 0) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _torch_distance_and_sweep_point_to_segment(
    px: torch.Tensor,
    py: torch.Tensor,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if eps is None:
        eps = torch.finfo(px.dtype).eps

    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    line_len_sq = dx * dx + dy * dy
    if line_len_sq == 0.0:
        dist = torch.sqrt((px - float(x1)) ** 2 + (py - float(y1)) ** 2 + eps)
        t = torch.zeros_like(dist)
        return dist, t

    t = ((px - float(x1)) * dx + (py - float(y1)) * dy) / line_len_sq
    t = torch.clamp(t, 0.0, 1.0)
    proj_x = float(x1) + t * dx
    proj_y = float(y1) + t * dy
    dist = torch.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2 + eps)
    return dist, t


def polygon_side_features_points_torch(
    coords_xy: torch.Tensor,
    vertices_ccw_np: np.ndarray,
    *,
    dist_scale: float = 1.0,
    sweep_centered: bool = True,
) -> torch.Tensor:
    """Return features [d1..dm, s1..sm] for polygon edges.

    d_i is distance to i-th edge segment; s_i is sweep parameter in [0,1]
    along that segment (projection clamped).

    If sweep_centered=True, map s_i from [0,1] -> [-1,1].
    """
    v = np.asarray(vertices_ccw_np, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
        raise ValueError("vertices_ccw_np must have shape (m,2) with m>=3")

    px = coords_xy[:, 0]
    py = coords_xy[:, 1]

    dists: List[torch.Tensor] = []
    sweeps: List[torch.Tensor] = []
    m = int(v.shape[0])
    for i in range(m):
        x1, y1 = float(v[i, 0]), float(v[i, 1])
        x2, y2 = float(v[(i + 1) % m, 0]), float(v[(i + 1) % m, 1])
        d, t = _torch_distance_and_sweep_point_to_segment(px, py, x1, y1, x2, y2)
        dists.append(d)
        sweeps.append(t)

    d = torch.stack(dists, dim=1) / float(dist_scale)
    s = torch.stack(sweeps, dim=1)
    if sweep_centered:
        s = 2.0 * s - 1.0

    return torch.cat([d, s], dim=1)


def _zero_init_last_linear(net: torch.nn.Module) -> None:
    for layer in reversed(list(net.modules())):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            return


def make_correction_net(
    *,
    net_type: str,
    in_dim: int,
    hidden: int,
    layers: int,
    first_omega_0: float = 30.0,
    hidden_omega_0: float = 30.0,
    kan_grid_size: int = 16,
    kan_grid_range: Tuple[float, float] = (-1.0, 1.0),
    kan_use_base: bool = True,
    kan_base_activation: str = "silu",
    device: str = "cpu",
) -> torch.nn.Module:
    t = str(net_type).strip().lower()
    arch = [int(in_dim)] + [int(hidden)] * int(layers) + [1]

    if t == "siren":
        net = network_defs.Siren(
            architecture=arch,
            outermost_linear=True,
            first_omega_0=float(first_omega_0),
            hidden_omega_0=float(hidden_omega_0),
        ).double().to(device)
        _zero_init_last_linear(net)
        return net

    if t in ("relu", "nn", "neuralnetwork"):
        net = network_defs.NeuralNetwork(architecture=arch).double().to(device)
        _zero_init_last_linear(net)
        return net

    if t in ("kan",):
        net = network_defs.KAN(
            architecture=arch,
            grid_size=int(kan_grid_size),
            grid_range=(float(kan_grid_range[0]), float(kan_grid_range[1])),
            use_base=bool(kan_use_base),
            base_activation=str(kan_base_activation),
        ).double().to(device)
        # KAN layers don't expose a final nn.Linear, so we cannot reuse
        # _zero_init_last_linear. Keep default init.
        return net

    raise ValueError(f"Unknown net_type={net_type!r}. Use 'SIREN', 'ReLU', or 'KAN'.")


def laplacian_via_autograd(scalar_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=scalar_field,
        inputs=coords,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2x = torch.autograd.grad(
        outputs=grad[:, 0],
        inputs=coords,
        grad_outputs=torch.ones_like(grad[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]

    d2y = torch.autograd.grad(
        outputs=grad[:, 1],
        inputs=coords,
        grad_outputs=torch.ones_like(grad[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1]

    return d2x + d2y


def discrete_laplacian_tensor_grid(u: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second-order FD Laplacian on a uniform tensor-product grid (physical coords).

    Returns an array with NaNs on the outermost boundary (where central stencils
    are not available).
    """
    u = np.asarray(u, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if u.ndim != 2 or x.shape != u.shape or y.shape != u.shape:
        raise ValueError("u, x, y must be 2D arrays with matching shapes")

    # Detect which axis corresponds to x variation (supports meshgrid indexing='ij' or 'xy').
    x_row = x[0, :]
    x_col = x[:, 0]
    y_row = y[0, :]
    y_col = y[:, 0]

    x_row_unique = np.unique(np.round(x_row, 12)).size
    x_col_unique = np.unique(np.round(x_col, 12)).size
    x_var_axis = 1 if x_row_unique > x_col_unique else 0

    if x_var_axis == 1:
        # x varies along axis=1, y varies along axis=0
        dx = float(np.mean(np.diff(x_row)))
        dy = float(np.mean(np.diff(y_col)))
        if not (dx > 0 and dy > 0):
            raise ValueError("Non-positive grid spacing detected")

        lap = np.full_like(u, np.nan, dtype=np.float64)
        d2y = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dy * dy)
        d2x = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dx * dx)
        lap[1:-1, 1:-1] = d2x + d2y
        return lap

    # x varies along axis=0, y varies along axis=1 (this is what collocation uses: indexing='ij').
    dx = float(np.mean(np.diff(x_col)))
    dy = float(np.mean(np.diff(y_row)))
    if not (dx > 0 and dy > 0):
        raise ValueError("Non-positive grid spacing detected")

    lap = np.full_like(u, np.nan, dtype=np.float64)
    d2x = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
    d2y = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    lap[1:-1, 1:-1] = d2x + d2y
    return lap


def _metrics_inside(u: np.ndarray, u_exact: np.ndarray, inside: np.ndarray) -> Tuple[float, float]:
    e = np.abs(u - u_exact)
    ei = e[inside]
    return float(np.mean(ei)), float(np.max(ei))


@dataclass(frozen=True)
class SweepConfig:
    net_type: str
    lr: float
    hidden: int
    layers: int
    iters: int
    batch_size: int
    boundary_alpha: float
    sample_beta: float
    corr_l2: float
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0


def train_residual_correction(
    *,
    u_base: np.ndarray,
    xP: np.ndarray,
    yP: np.ndarray,
    inside: np.ndarray,
    weight_model: torch.nn.Module,
    vertices_ccw_np: np.ndarray,
    f_phys: ArrayFn,
    u_exact_phys: Optional[ArrayFn] = None,
    dist_scale: float = 2.0,
    w_min: float = 1e-6,
    boundary_alpha: float = 120.0,
    sample_beta: float = 120.0,
    net_type: str = "SIREN",
    hidden: int = 64,
    layers: int = 2,
    first_omega_0: float = 30.0,
    hidden_omega_0: float = 30.0,
    iters: int = 200,
    batch_size: int = 128,
    lr: float = 1e-4,
    corr_l2: float = 1e-6,
    grad_clip: float = 1.0,
    eval_every: int = 25,
    eval_n: int = 2000,
    select_by: str = "linf_mae",
    device: str = "cpu",
    verbose_every: int = 100,
) -> Dict:
    """Train correction on collocation tensor grid in physical coordinates."""

    u_base = np.asarray(u_base, dtype=np.float64)
    xP = np.asarray(xP, dtype=np.float64)
    yP = np.asarray(yP, dtype=np.float64)
    inside = np.asarray(inside, dtype=bool)

    lap_base = discrete_laplacian_tensor_grid(u_base, xP, yP)

    # Candidates: inside, not too close to outer grid boundary, and lap_base is finite.
    mask = inside.copy()
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask &= np.isfinite(lap_base)

    # Also avoid very tiny w (we can still keep boundary layer but reduce numerical noise)
    with torch.no_grad():
        crd_all = torch.tensor(
            np.stack([xP[mask], yP[mask]], axis=1), dtype=torch.float64, device=device
        )
        w_all = weight_model(crd_all).view(-1).detach().cpu().numpy()
    mask_flat = w_all >= float(w_min)

    idx_all = np.argwhere(mask)
    if idx_all.shape[0] == 0:
        raise RuntimeError("No interior points available for correction training")

    idx_all = idx_all[mask_flat]
    if idx_all.shape[0] < max(32, int(batch_size)):
        # fall back: drop w_min filtering
        idx_all = np.argwhere(mask)

    # sampling weights, biased to small w (near boundary)
    if float(sample_beta) > 0.0:
        with torch.no_grad():
            crd_w = torch.tensor(
                np.stack([xP[idx_all[:, 0], idx_all[:, 1]], yP[idx_all[:, 0], idx_all[:, 1]]], axis=1),
                dtype=torch.float64,
                device=device,
            )
            w_s = torch.clamp(weight_model(crd_w).view(-1), min=0.0).detach().cpu().numpy()
        pw = np.exp(-float(sample_beta) * w_s)
        pw = pw / np.sum(pw)
    else:
        pw = None

    n_sides = int(np.asarray(vertices_ccw_np).shape[0])
    phi = make_correction_net(
        net_type=net_type,
        in_dim=2 * n_sides,
        hidden=hidden,
        layers=layers,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        device=device,
    )
    opt = torch.optim.Adam(phi.parameters(), lr=float(lr))

    # deterministic eval set
    best_state = None
    best_key = None

    if u_exact_phys is not None:
        # compute exact on the grid once
        u_exact_grid = np.asarray(u_exact_phys(xP, yP), dtype=np.float64)

        eval_idx = np.argwhere(inside)
        if int(eval_n) > 0 and eval_idx.shape[0] > int(eval_n):
            step = max(1, int(eval_idx.shape[0] // int(eval_n)))
            eval_idx = eval_idx[::step][: int(eval_n)]

        ey = eval_idx[:, 0]
        ex = eval_idx[:, 1]
        u_base_eval = u_base[ey, ex]
        u_exact_eval = u_exact_grid[ey, ex]

        coords_eval = torch.tensor(
            np.stack([xP[ey, ex], yP[ey, ex]], axis=1),
            dtype=torch.float64,
            device=device,
        )
        with torch.no_grad():
            feats_eval = polygon_side_features_points_torch(
                coords_eval, vertices_ccw_np, dist_scale=dist_scale, sweep_centered=True
            )
    else:
        u_exact_grid = None
        coords_eval = None
        feats_eval = None

    def key_from_metrics(mae: float, linf: float) -> Tuple[float, float]:
        if str(select_by).lower() == "mae_linf":
            return (mae, linf)
        return (linf, mae)

    def make_corr(coords_in: torch.Tensor, feats_in: torch.Tensor) -> torch.Tensor:
        w_t = weight_model(coords_in).view(-1)
        w_pos = torch.clamp(w_t, min=0.0)
        phi_val = phi(feats_in).view(-1)
        env = torch.exp(-float(boundary_alpha) * w_pos) if float(boundary_alpha) > 0.0 else 1.0
        return w_pos * env * phi_val

    for it in range(1, int(iters) + 1):
        sel = np.random.choice(idx_all.shape[0], size=int(batch_size), replace=False, p=pw)
        pts = idx_all[sel]
        iy = pts[:, 0]
        ix = pts[:, 1]

        xs = xP[iy, ix]
        ys = yP[iy, ix]
        lap_u = lap_base[iy, ix]
        f_np = np.asarray(f_phys(xs, ys), dtype=np.float64).reshape(-1)

        coords = torch.tensor(
            np.stack([xs, ys], axis=1), dtype=torch.float64, device=device, requires_grad=True
        )
        feats = polygon_side_features_points_torch(coords, vertices_ccw_np, dist_scale=dist_scale, sweep_centered=True)

        lap_u_t = torch.tensor(lap_u, dtype=torch.float64, device=device)
        f_t = torch.tensor(f_np, dtype=torch.float64, device=device)

        corr = make_corr(coords, feats)
        lap_corr = laplacian_via_autograd(corr, coords)

        resid = -(lap_u_t + lap_corr) - f_t
        loss = torch.mean(torch.abs(resid))
        if float(corr_l2) > 0.0:
            loss = loss + float(corr_l2) * torch.mean(corr * corr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(phi.parameters(), max_norm=float(grad_clip))
        opt.step()

        if verbose_every and (it % int(verbose_every) == 0 or it == 1 or it == int(iters)):
            print(f"it={it:4d}/{iters} | L1(resid)={loss.item():.3e}")

        if coords_eval is not None and eval_every and (it % int(eval_every) == 0 or it == 1 or it == int(iters)):
            with torch.no_grad():
                corr_eval = make_corr(coords_eval, feats_eval).cpu().numpy()
            u_pred = u_base_eval + corr_eval
            e = np.abs(u_pred - u_exact_eval)
            mae = float(np.mean(e))
            linf = float(np.max(e))
            k = key_from_metrics(mae, linf)
            if best_key is None or k < best_key:
                best_key = k
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in phi.state_dict().items()}

    if best_state is not None:
        phi.load_state_dict(best_state)

    # Full-grid correction
    Xf = torch.tensor(xP.reshape(-1), dtype=torch.float64, device=device)
    Yf = torch.tensor(yP.reshape(-1), dtype=torch.float64, device=device)
    pts_full = torch.stack([Xf, Yf], dim=1)
    with torch.no_grad():
        feats_full = polygon_side_features_points_torch(
            pts_full, vertices_ccw_np, dist_scale=dist_scale, sweep_centered=True
        )
        corr_full = make_corr(pts_full, feats_full).view_as(torch.tensor(xP, dtype=torch.float64, device=device))

    corr_grid = corr_full.detach().cpu().numpy()
    u_corr = u_base + corr_grid

    out: Dict = {
        "phi": phi,
        "corr": corr_grid,
        "u_corrected": u_corr,
        "best_key": best_key,
    }
    if u_exact_grid is not None:
        out["u_exact_grid"] = u_exact_grid
        out["metrics"] = {
            "base": _metrics_inside(u_base, u_exact_grid, inside),
            "corr": _metrics_inside(u_corr, u_exact_grid, inside),
        }
    return out


def run_guarded_sweep(
    *,
    u_base: np.ndarray,
    xP: np.ndarray,
    yP: np.ndarray,
    inside: np.ndarray,
    weight_model: torch.nn.Module,
    vertices_ccw_np: np.ndarray,
    f_phys: ArrayFn,
    u_exact_phys: Optional[ArrayFn],
    dist_scale: float,
    guard_mae_frac: float = 0.05,
    guard_mae_abs: float = 0.0,
    select_by: str = "linf_mae",
    sweep: Sequence[SweepConfig],
    seed: int = 0,
    device: str = "cpu",
) -> Dict:
    set_seed(seed)

    u_exact_grid = np.asarray(u_exact_phys(xP, yP), dtype=np.float64) if u_exact_phys is not None else None
    base_mae, base_linf = _metrics_inside(u_base, u_exact_grid, inside) if u_exact_grid is not None else (np.nan, np.nan)

    best_overall: Optional[Dict] = None
    best_pair_overall: Optional[Tuple[float, float]] = None
    best_cfg_overall: Optional[SweepConfig] = None

    # Guarded best starts as baseline so we never return a worse-than-baseline
    # candidate that merely "passes" the MAE constraint.
    best_guarded: Optional[Dict] = None
    best_pair_guarded: Optional[Tuple[float, float]] = None
    best_cfg_guarded: Optional[SweepConfig] = None

    trials: List[Dict] = []

    pair_base = (float(base_linf), float(base_mae)) if str(select_by).lower() == "linf_mae" else (float(base_mae), float(base_linf))
    # Baseline record (always guarded)
    corr0 = np.zeros_like(u_base, dtype=np.float64)
    baseline_out: Dict = {
        "phi": None,
        "corr": corr0,
        "u_corrected": np.asarray(u_base, dtype=np.float64),
        "best_key": pair_base,
        "selected_cfg": None,
        "trials": trials,
        "base_metrics": {"mae": float(base_mae), "linf": float(base_linf)},
        "used_baseline": True,
    }
    if u_exact_grid is not None:
        baseline_out["u_exact_grid"] = u_exact_grid
        baseline_out["metrics"] = {"base": (float(base_mae), float(base_linf)), "corr": (float(base_mae), float(base_linf))}

    # Initialize guarded best to baseline.
    best_guarded = baseline_out
    best_pair_guarded = pair_base
    best_cfg_guarded = None

    for cfg in sweep:
        out = train_residual_correction(
            u_base=u_base,
            xP=xP,
            yP=yP,
            inside=inside,
            weight_model=weight_model,
            vertices_ccw_np=vertices_ccw_np,
            f_phys=f_phys,
            u_exact_phys=u_exact_phys,
            dist_scale=dist_scale,
            boundary_alpha=float(cfg.boundary_alpha),
            sample_beta=float(cfg.sample_beta),
            net_type=str(cfg.net_type),
            hidden=int(cfg.hidden),
            layers=int(cfg.layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
            iters=int(cfg.iters),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            corr_l2=float(cfg.corr_l2),
            select_by=str(select_by),
            device=device,
            verbose_every=0,
        )

        if u_exact_grid is None:
            # no manufactured exact => cannot select safely
            return out

        mae, linf = _metrics_inside(out["u_corrected"], u_exact_grid, inside)
        mae_ratio = float(mae) / max(float(base_mae), 1e-300)
        linf_ratio = float(linf) / max(float(base_linf), 1e-300)
        pair = (linf, mae) if str(select_by).lower() == "linf_mae" else (mae, linf)

        guard_limit = float(base_mae) + max(float(guard_mae_abs), float(base_mae) * float(guard_mae_frac))
        passed = bool(float(mae) <= guard_limit)
        trials.append(
            {
                "cfg": cfg,
                "mae": float(mae),
                "linf": float(linf),
                "mae_ratio": mae_ratio,
                "linf_ratio": linf_ratio,
                "passed_guard": passed,
                "guard_limit": guard_limit,
            }
        )

        if best_pair_overall is None or pair < best_pair_overall:
            best_pair_overall = pair
            best_overall = out
            best_cfg_overall = cfg

        if passed:
            if best_pair_guarded is None or pair < best_pair_guarded:
                best_pair_guarded = pair
                best_guarded = out
                best_cfg_guarded = cfg

    # Always return guarded best (baseline is the initial guarded best).
    if best_guarded is None:
        best_guarded = baseline_out
        best_cfg_guarded = None

    # Annotate and return.
    best_guarded["selected_cfg"] = best_cfg_guarded
    best_guarded["trials"] = trials
    best_guarded["base_metrics"] = {"mae": float(base_mae), "linf": float(base_linf)}
    best_guarded["used_baseline"] = bool(best_cfg_guarded is None)
    return best_guarded
