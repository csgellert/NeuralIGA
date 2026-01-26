# Neural Analysis Folder Guide

## Scope and PDE model
- Implements immersed-domain solvers for Poisson problems of the form $-\Delta u = f$ with Dirichlet data enforced implicitly by a signed distance / weight function \(w(x,y)\).
- Domains are given as zero level sets of analytic or neural SDFs; the weight function is \(w = \max(0, d(x,y))\) (or transformed variants) so interior points satisfy \(w>0\).
- Primary formulations:
  - Weighted B-spline Galerkin (standard immersed IGA)
  - WEB-splines (Weighted Extended B-splines) for conditioning improvements
  - WEB-spline collocation on structured grids

## Solver entry points
- [Neural Analysis/main.py](Neural%20Analysis/main.py): quick driver to assemble and solve a single case. Supports standard weighted B-splines, full WEB transform, or selective diagonal extraction. Uses parameters `DIVISIONS`, `ORDER`, `DELTA`, and flags `USE_WEB`, `USE_WEB_TRANSFORM`, `USE_WEB_DIAG_EXTRACT`.
- [Neural Analysis/analyser.py](Neural%20Analysis/analyser.py): sweeps Galerkin runs over orders/divisions, optional WEB, saves JSON metrics via `save_simulation_results`.
- [Neural Analysis/analyser_collocation.py](Neural%20Analysis/analyser_collocation.py): convergence sweeps for WEB-spline collocation; configures function cases, degree `n`, grid `H`, and domain bounds.
- [Neural Analysis/bench_web_vs_std.py](Neural%20Analysis/bench_web_vs_std.py): small benchmark comparing assembly/solve times and errors for standard vs WEB.
- Notebooks (e.g., [Neural Analysis/colloc_results.ipynb](Neural%20Analysis/colloc_results.ipynb)) demonstrate interactive runs and visualization.

## Galerkin solvers (weighted B-splines)
- [Neural Analysis/FEM.py](Neural%20Analysis/FEM.py): core weighted B-spline Galerkin assembly for Poisson. Handles Gauss quadrature, adaptive subdivision near the boundary, right-hand-side/Dirichlet data (`FUNCTION_CASE`), and solves with `solveWeak`. Uses implicit geometry via `mesh.distance_with_derivative`.
- [Neural Analysis/evaluation.py](Neural%20Analysis/evaluation.py): error metrics (MSE, MAE, L\_inf, H1), random-point sampling, heatmaps/contours for solutions, and pretty-printers.
- [Neural Analysis/utility.py](Neural%20Analysis/utility.py): small visualization helper for recursive element subdivision (debugging quadrature layout).

## WEB-spline variants
- [Neural Analysis/FEM_WEB.py](Neural%20Analysis/FEM_WEB.py): paper-faithful WEB implementation (Hollig/Reif/Wipper). Classifies inner/outer B-splines from element types, builds extension coefficients (via Hollig-style Lagrange polynomials), assembles reduced WEB system, and provides transforms from standard systems.
- [Neural Analysis/hollig_exact_extension.py](Neural%20Analysis/hollig_exact_extension.py): exact computation of extension coefficients (strict or fallback LS/nearest-neighbor options).
- [Neural Analysis/evaluation_WEB.py](Neural%20Analysis/evaluation_WEB.py): reconstruction in WEB basis, detailed per-point contribution breakdown, accuracy metrics, and visualization utilities.

## Collocation solver (WEB)
- [Neural Analysis/collocation_WEB.py](Neural%20Analysis/collocation_WEB.py): full WEB collocation pipeline for \(-\Delta u = f\) on implicitly defined domains. Provides:
  - Function/BC library (`FUNCTION_CASE`) with exact solutions and derivatives.
  - Cardinal B-spline evaluation/derivatives and extension coefficient calculation.
  - Weight-function wrapper `NeuralWeightFunction` and `create_domain_transformer` for mapping arbitrary rectangular domains to solver coordinates.
  - Assembly of collocation matrices, conditioning stats, gradient recovery, and visualization helpers.
- Collocation analysis and plotting helpers live in [Neural Analysis/analyser_collocation.py](Neural%20Analysis/analyser_collocation.py) and notebooks such as [Neural Analysis/visualize_collocation_convergence.ipynb](Neural%20Analysis/visualize_collocation_convergence.ipynb).

## Geometry, SDFs, and meshes
- Analytic SDFs for PDE runs: [Neural Analysis/Geomertry.py](Neural%20Analysis/Geomertry.py) defines circle and L-shape distance networks plus Bspline wrappers; also reused to seed WEB weight functions.
- Structured mesh + distance utilities: [Neural Analysis/mesh.py](Neural%20Analysis/mesh.py) builds rectangular knot grids, distance transforms (`distanceFromContur`, `distance_with_derivative`), optional sigmoid/tanh transforms, and plotting helpers.
- Generic SDF library: [Neural Analysis/SDF.py](Neural%20Analysis/SDF.py) supplies vectorized distances for lines, stars, L-shape, ray-casting inside checks, and sampling helpers.
- B-spline geometry toolkit: [Neural Analysis/geometry_bspline.py](Neural%20Analysis/geometry_bspline.py) and [Neural Analysis/geometry_definitions.py](Neural%20Analysis/geometry_definitions.py) generate control points for circles/stars/polygons/L-shape, evaluate basis/derivatives/normals/curvature, and compute signed distances to B-spline curves.
- Visualization of geometries and error maps: [Neural Analysis/geometry_visualisation.py](Neural%20Analysis/geometry_visualisation.py) plots curves, normals, distance fields, and model error maps.

## Neural implicit models (weight functions)
- Network architectures: [Neural Analysis/network_defs.py](Neural%20Analysis/network_defs.py) provides ReLU MLPs, SIREN variants (sine activations), mixed SIRELU, and loaders for testing pretrained SDF models.
- Training/data generation: [Neural Analysis/NeuralImplicit.py](Neural%20Analysis/NeuralImplicit.py) builds datasets (analytic shapes or B-spline contours via `SDF`/`geometry_bspline`), trains SDF networks, and includes plotting/animation utilities.
- Stored artifacts: pretrained checkpoints under [Neural Analysis/trained_models](Neural%20Analysis/trained_models) and saved simulation JSON under [Neural Analysis/simulation_results](Neural%20Analysis/simulation_results).

## Typical workflows
- **Single run (Galerkin or WEB)**: adjust flags in [Neural Analysis/main.py](Neural%20Analysis/main.py) (choose `USE_WEB` or transform options), set `FUNCTION_CASE`/`DOMAIN` in [Neural Analysis/FEM.py](Neural%20Analysis/FEM.py) if needed, and execute. Metrics/plots come from `evaluation` or `evaluation_WEB`.
- **Collocation study**: use [Neural Analysis/analyser_collocation.py](Neural%20Analysis/analyser_collocation.py) to sweep degrees and grids; notebook [Neural Analysis/colloc_results.ipynb](Neural%20Analysis/colloc_results.ipynb) shows an example with domain transforms and error computation.
- **Compare standard vs WEB**: run [Neural Analysis/bench_web_vs_std.py](Neural%20Analysis/bench_web_vs_std.py) for quick timing/error comparison.
- **Train or test SDF networks**: generate data/train in [Neural Analysis/NeuralImplicit.py](Neural%20Analysis/NeuralImplicit.py); consume models via `load_test_model` in [Neural Analysis/network_defs.py](Neural%20Analysis/network_defs.py) or directly instantiate analytic distances in [Neural Analysis/Geomertry.py](Neural%20Analysis/Geomertry.py).

## Notes and conventions
- Default domain bounds and function cases are defined in the solver modules; adjust `FUNCTION_CASE` to switch manufactured solutions (circle, L-shape, double circle, etc.).
- All solvers default to `torch.float64` for stability; collocation and WEB routines expect consistent dtype.
- WEB transforms rely on stable inner/outer classification; see [Neural Analysis/FEM_WEB.py](Neural%20Analysis/FEM_WEB.py) and [Neural Analysis/hollig_exact_extension.py](Neural%20Analysis/hollig_exact_extension.py) for the precise algorithm.
- Visualization notebooks in this folder read results from `simulation_results` and `trained_models`; keep paths relative when moving the project.
