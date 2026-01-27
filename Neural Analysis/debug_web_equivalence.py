"""Debug equivalence between direct WEB assembly and transform-to-WEB.

This script checks whether the reduced matrices/vectors match:
  K_web_direct  ?=  E K_full E^T
  F_web_direct  ?=  E F_full

Run from repo root:
  .venv/Scripts/python.exe "Neural Analysis/debug_web_equivalence.py"

Note: this is a diagnostic; it does not write any result files.
"""

import argparse
import time

import numpy as np
import torch

import mesh
import FEM
import FEM_WEB
import Geomertry


torch.set_default_dtype(torch.float64)


def _rel_fro_norm(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main():
    parser = argparse.ArgumentParser(description="Check equivalence: direct WEB vs transformed standard system")
    parser.add_argument("--div", type=int, default=6, help="Number of divisions (default: 6)")
    parser.add_argument("--order", type=int, default=2, help="Spline degree p=q (default: 2)")
    parser.add_argument("--delta", type=float, default=0.005, help="Mesh delta (default: 0.005)")
    parser.add_argument(
        "--max-subdiv",
        type=int,
        default=1,
        help="Override MAX_SUBDIVISION for speed (default: 1)",
    )
    parser.add_argument(
        "--web-norm",
        action="store_true",
        help="Enable WEB normalization (Definition 2): scale by 1/w(x_i)",
    )
    args = parser.parse_args()

    DIVISIONS = int(args.div)
    ORDER = int(args.order)
    DELTA = float(args.delta)

    # Make the run fast and ensure BOTH assembly paths use the same subdivision depth.
    FEM.MAX_SUBDIVISION = int(args.max_subdiv)
    FEM_WEB.MAX_SUBDIVISION = int(args.max_subdiv)

    model = Geomertry.AnaliticalDistanceCircle()

    default = mesh.getDefaultValues(div=DIVISIONS, order=ORDER, delta=DELTA)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knot_u, knot_v, weights, ctrlpts = mesh.generateRectangularMesh(*default)

    assert p == q
    assert xDivision == yDivision

    n_total = (xDivision + p + 1) * (yDivision + q + 1)

    print("=" * 80)
    print(f"Equivalence check: div={xDivision}, p=q={p}, total DOFs={n_total}")
    print("=" * 80)

    # 1) Full standard weighted system
    K_full = np.zeros((n_total, n_total), dtype=np.float64)
    F_full = np.zeros(n_total, dtype=np.float64)
    t0 = time.time()
    K_full, F_full, etype_full = FEM.processAllElements(
        model, p, q, knot_u, knot_v, xDivision, yDivision, K_full, F_full
    )
    print(f"Standard assembly time: {time.time() - t0:.2f}s")

    # 2) Transform to WEB (build E and reduced K/F)
    t0 = time.time()
    K_red_from_transform, F_red_from_transform, etype_t, bsp_t, ext_basis_t, E = (
        FEM_WEB.transformStandardSystemToWEB(
            K_full,
            F_full,
            model,
            p,
            q,
            knot_u,
            knot_v,
            xDivision,
            yDivision,
            extension_strict=True,
            web_use_weight_normalization=bool(args.web_norm),
        )
    )
    print(f"Transform build+apply time: {time.time() - t0:.2f}s")

    # 3) Direct WEB reduced assembly
    t0 = time.time()
    K_web_direct, F_web_direct, etype_web, bsp_web, ext_basis_web = FEM_WEB.processAllElementsWEB(
        model,
        p,
        q,
        knot_u,
        knot_v,
        xDivision,
        yDivision,
        extension_strict=True,
        web_use_weight_normalization=bool(args.web_norm),
    )
    print(f"Direct WEB assembly time: {time.time() - t0:.2f}s")

    # 4) Compare inner basis ordering
    inner_t = bsp_t["inner"]
    inner_w = bsp_web["inner"]
    same_inner_order = inner_t == inner_w
    print(f"Inner DOFs (transform): {len(inner_t)}")
    print(f"Inner DOFs (direct WEB): {len(inner_w)}")
    print(f"Inner ordering identical: {same_inner_order}")

    if len(inner_t) != len(inner_w):
        print("WARNING: n_inner differs; methods are not comparable as-is.")
        return

    # If ordering differs, build a permutation to align direct WEB to transform ordering
    if not same_inner_order:
        idx_map = {idx: k for k, idx in enumerate(inner_w)}
        perm = np.array([idx_map[idx] for idx in inner_t], dtype=int)
        K_web_aligned = K_web_direct[np.ix_(perm, perm)]
        F_web_aligned = F_web_direct[perm]
    else:
        K_web_aligned = K_web_direct
        F_web_aligned = F_web_direct

    # 5) Compare reduced matrices/vectors
    print("-" * 80)
    print("Matrix/vector equivalence:")
    print(f"rel ||K_web - K_trans||_F  = {_rel_fro_norm(K_web_aligned, K_red_from_transform):.3e}")
    print(f"max |K_web - K_trans|      = {_max_abs(K_web_aligned, K_red_from_transform):.3e}")
    print(f"rel ||F_web - F_trans||_2  = {_rel_fro_norm(F_web_aligned, F_red_from_transform):.3e}")
    print(f"max |F_web - F_trans|      = {_max_abs(F_web_aligned, F_red_from_transform):.3e}")

    # 6) Solve both reduced systems and compare coefficients
    u_t = FEM_WEB.solveWEB(K_red_from_transform, F_red_from_transform)
    u_w = FEM_WEB.solveWEB(K_web_aligned, F_web_aligned)

    print("-" * 80)
    print("Solution coefficient comparison (reduced space):")
    rel_u = _rel_fro_norm(u_w, u_t)
    max_u = _max_abs(u_w, u_t)
    print(f"rel ||u_web - u_trans||_2  = {rel_u:.3e}")
    print(f"max |u_web - u_trans|      = {max_u:.3e}")

    # 7) Optional: back-transform transform-solution to full space for sanity
    u_full_from_transform = E.T @ u_t
    print("-" * 80)
    print("Sanity:")
    print(f"u_full_from_transform finite: {np.isfinite(u_full_from_transform).all()}")


if __name__ == "__main__":
    main()
