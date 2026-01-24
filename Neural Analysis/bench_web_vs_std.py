import time
import numpy as np
import torch

import mesh
import FEM
import FEM_WEB
import evaluation
import evaluation_WEB
from Geomertry import AnaliticalDistanceCircle


class _NoTqdm:
    def __init__(self, iterable=None, total=None, desc=None, unit=None):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        return None

    def set_description(self, desc=None):
        return None


def run(divisions: int = 6, order: int = 1, delta: float = 0.005, n_eval: int = 200, seed: int = 0, max_subdivision: int = 1):
    torch.set_default_dtype(torch.float64)

    model = AnaliticalDistanceCircle()

    default = mesh.getDefaultValues(div=divisions, order=order, delta=delta)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knot_u, knot_w, weights, ctrlpts = mesh.generateRectangularMesh(*default)

    print(f"div={xDivision} p={p} q={q} n_eval={n_eval} max_subdivision={max_subdivision}")

    # Silence progress bars for cleaner benchmarking output
    FEM.tqdm = _NoTqdm
    FEM_WEB.tqdm = _NoTqdm

    # Make the quick benchmark finish fast
    FEM.MAX_SUBDIVISION = max_subdivision
    FEM_WEB.MAX_SUBDIVISION = max_subdivision

    # Standard weighted B-splines
    n_dof_std = (xDivision + p + 1) * (yDivision + q + 1)
    K_std = np.zeros((n_dof_std, n_dof_std))
    F_std = np.zeros(n_dof_std)

    t0 = time.time()
    K_std, F_std, etype_std = FEM.processAllElements(
        model, p, q, knot_u, knot_w, xDivision, yDivision, K_std, F_std
    )
    t1 = time.time()
    u_std = FEM.solveWeak(K_std, F_std)
    t2 = time.time()

    metrics_std = evaluation.evaluateAccuracy(model, u_std, p, q, knot_u, knot_w, N=n_eval, seed=seed)

    print("\n[STANDARD]")
    print("etype", etype_std)
    print("assembly_s", round(t1 - t0, 4), "solve_s", round(t2 - t1, 4))
    print("MAE", metrics_std["MAE"], "H1_full", metrics_std["H1_full"], "valid", metrics_std["n_valid_points"])

    # WEB
    t0 = time.time()
    K_web, F_web, etype_web, bsp_class, ext_basis = FEM_WEB.processAllElementsWEB(
        model, p, q, knot_u, knot_w, xDivision, yDivision
    )
    t1 = time.time()
    u_web = FEM_WEB.solveWEB(K_web, F_web)
    t2 = time.time()

    metrics_web = evaluation_WEB.evaluateAccuracyWEB(
        model, u_web, p, q, knot_u, knot_w, bsp_class, ext_basis, N=n_eval, seed=seed
    )

    print("\n[WEB]")
    print("etype", etype_web)
    print("n_inner", bsp_class["n_inner"], "n_outer", bsp_class["n_outer"])
    print("assembly_s", round(t1 - t0, 4), "solve_s", round(t2 - t1, 4))
    print("MAE", metrics_web["MAE"], "H1_full", metrics_web["H1_full"], "valid", metrics_web["n_valid_points"])


if __name__ == "__main__":
    run()
