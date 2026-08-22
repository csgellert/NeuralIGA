"""Lightning solver for the 2-D Laplace equation on polygonal domains.

This module is a focused Python port of the solver in ``lightning/laplace.m``.
It contains the numerical solver only: no examples, plotting, or convergence
study.  The public case-11 helper uses the same pentagon and boundary data as
``PDE_testcases`` and returns a callable solution suitable for WEB workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.linalg import lstsq


ArrayLike = Union[np.ndarray, Sequence[float], float]
BoundaryFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
BoundaryData = Union[BoundaryFunction, Sequence[Union[BoundaryFunction, float]]]


@dataclass
class LightningLaplaceSolution:
    """Callable harmonic solution and diagnostics returned by the solver."""

    vertices: np.ndarray
    center: complex
    scale: float
    polynomial_coefficients: np.ndarray
    pole_coefficients: np.ndarray
    poles: np.ndarray
    pole_scales: np.ndarray
    arnoldi_hessenberg: Optional[np.ndarray]
    degree: int
    max_error: float
    iterations: int
    boundary_points: np.ndarray
    boundary_values: np.ndarray

    def _evaluate_complex(self, z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).ravel()
        zz = np.concatenate(([self.center], flat))

        if self.arnoldi_hessenberg is None:
            q = ((zz - self.center) / self.scale)[:, None] ** np.arange(self.degree + 1)
        else:
            hessenberg = self.arnoldi_hessenberg
            q_columns = [np.ones(zz.size, dtype=np.complex128)]
            for column in range(self.degree):
                value = (zz - self.center) * q_columns[column]
                for previous in range(column + 1):
                    value -= hessenberg[previous, column] * q_columns[previous]
                denominator = hessenberg[column + 1, column]
                q_columns.append(value / denominator)
            q = np.column_stack(q_columns)

        basis = q
        if self.poles.size:
            basis = np.column_stack(
                (basis, self.pole_scales[None, :] / (zz[:, None] - self.poles[None, :]))
            )

        values = basis @ np.concatenate((self.polynomial_coefficients, self.pole_coefficients))
        values = values[1:] - 1j * values[0].imag
        return values.reshape(np.asarray(z).shape)

    def _evaluate_complex_and_derivative(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (f(z), f'(z)) for the underlying holomorphic function f, with
        u(x,y) = Re(f(z)) the harmonic solution.

        Every basis piece here (polynomial/Vandermonde, Arnoldi, and pole
        terms) is an explicit, elementary function of z, so f' is built
        analytically term by term -- no finite differences or autograd
        needed. The Arnoldi basis is defined by a linear recurrence in z, so
        its derivative satisfies the same recurrence differentiated by the
        product rule, carried alongside the values themselves.
        """
        flat = np.asarray(z, dtype=np.complex128).ravel()
        zz = np.concatenate(([self.center], flat))

        if self.arnoldi_hessenberg is None:
            zc = (zz - self.center) / self.scale
            powers = np.arange(self.degree + 1)
            q = zc[:, None] ** powers
            dq = np.zeros_like(q)
            if self.degree >= 1:
                dq[:, 1:] = (powers[None, 1:] * zc[:, None] ** (powers[None, 1:] - 1)) / self.scale
        else:
            hessenberg = self.arnoldi_hessenberg
            q_columns = [np.ones(zz.size, dtype=np.complex128)]
            dq_columns = [np.zeros(zz.size, dtype=np.complex128)]
            for column in range(self.degree):
                value = (zz - self.center) * q_columns[column]
                dvalue = q_columns[column] + (zz - self.center) * dq_columns[column]
                for previous in range(column + 1):
                    value -= hessenberg[previous, column] * q_columns[previous]
                    dvalue -= hessenberg[previous, column] * dq_columns[previous]
                denominator = hessenberg[column + 1, column]
                q_columns.append(value / denominator)
                dq_columns.append(dvalue / denominator)
            q = np.column_stack(q_columns)
            dq = np.column_stack(dq_columns)

        basis = q
        dbasis = dq
        if self.poles.size:
            diff = zz[:, None] - self.poles[None, :]
            basis = np.column_stack((basis, self.pole_scales[None, :] / diff))
            dbasis = np.column_stack((dbasis, -self.pole_scales[None, :] / diff**2))

        coefficients = np.concatenate((self.polynomial_coefficients, self.pole_coefficients))
        values = basis @ coefficients
        derivative = dbasis @ coefficients

        # The branch-fixing shift subtracted from `values` is a real constant
        # (independent of z), so it has zero derivative and is omitted here.
        values = values[1:] - 1j * values[0].imag
        derivative = derivative[1:]
        shape = np.asarray(z).shape
        return values.reshape(shape), derivative.reshape(shape)

    def evaluate(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """Evaluate the real harmonic solution at complex or Cartesian points."""
        if y is None:
            z = np.asarray(x, dtype=np.complex128)
        else:
            x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            z = x_arr + 1j * y_arr
        return np.real(self._evaluate_complex(z))

    def evaluate_gradient(
        self, x: ArrayLike, y: Optional[ArrayLike] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (u, du/dx, du/dy) for the real harmonic solution.

        Exact via the Cauchy-Riemann equations: for holomorphic f = u + iv,
        f'(z) = du/dx + i*dv/dx = du/dx - i*du/dy, so du/dx = Re(f') and
        du/dy = -Im(f'). No finite differences are used.
        """
        if y is None:
            z = np.asarray(x, dtype=np.complex128)
        else:
            x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            z = x_arr + 1j * y_arr
        f, fprime = self._evaluate_complex_and_derivative(z)
        return np.real(f), np.real(fprime), -np.imag(fprime)

    def laplacian(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """Laplacian of the harmonic solution.

        Identically zero: u = Re(f) for holomorphic f solves Delta(u) = 0
        exactly by construction (real/imaginary parts of a holomorphic
        function are harmonic), independent of the least-squares fit
        residual at the boundary.
        """
        shape = np.broadcast_shapes(np.shape(x), np.shape(y) if y is not None else ())
        return np.zeros(shape, dtype=float)

    def __call__(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        return self.evaluate(x, y)


def _as_ccw_vertices(vertices: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (vertices, was_reversed) with vertices in CCW order.

    ``was_reversed`` lets callers keep any per-vertex side array (e.g. a
    corner mask) consistent with the possibly-flipped vertex order.
    """
    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 3:
        raise ValueError("vertices must have shape (number_of_vertices, 2)")
    area2 = np.sum(values[:, 0] * np.roll(values[:, 1], -1) - values[:, 1] * np.roll(values[:, 0], -1))
    if area2 >= 0:
        return values, False
    return values[::-1].copy(), True


def _polygon_geometry(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, complex]:
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    edges = ends - starts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("polygon edges must have non-zero length")
    tangents = edges / lengths[:, None]
    outward = tangents[:, 1] - 1j * tangents[:, 0]
    real_vertices = vertices[:, 0] + 1j * vertices[:, 1]
    center = complex(np.mean(vertices[:, 0]), np.mean(vertices[:, 1]))
    scale = float(np.max(np.ptp(vertices, axis=0)))
    return real_vertices, lengths, outward, scale, center


def _inside_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Point-in-polygon test valid for convex *or* concave simple polygons
    (even-odd ray-casting rule), vectorized over ``points``.

    The pole-placement loop below only wants to know which candidate pole
    positions fall outside the polygon; a half-plane-intersection test would
    silently be wrong here for a non-convex polygon (e.g. case 12's mixed
    pentagon, which bulges inward on 3 sides) -- it would place poles
    *inside* the domain, causing the solution to blow up near them.
    """
    x = points[:, 0]
    y = points[:, 1]
    xi = vertices[:, 0]
    yi = vertices[:, 1]
    xj = np.roll(xi, -1)
    yj = np.roll(yi, -1)
    eps = 1e-14
    crosses = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])) & (
        x[:, None] < (xj[None, :] - xi[None, :]) * (y[:, None] - yi[None, :]) / (yj[None, :] - yi[None, :] + eps) + xi[None, :]
    )
    return (np.count_nonzero(crosses, axis=1) % 2) == 1


def _boundary_samples(
    vertices: np.ndarray,
    side_lengths: np.ndarray,
    pole_counts: np.ndarray,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    tangents = []
    sides = []
    for side, (start, end, length, count) in enumerate(
        zip(vertices, np.roll(vertices, -1, axis=0), side_lengths, pole_counts)
    ):
        cluster = np.exp(4.0 * (np.sqrt(np.arange(1, count + 1)) - np.sqrt(count)))
        cluster = cluster[cluster > 1e-15]
        distances = np.concatenate((cluster / 3.0, 2.0 * cluster / 3.0))
        parameters = np.concatenate((distances[distances < length], np.linspace(0.0, length, max(minimum_samples, count))))
        previous_side = (side - 1) % len(vertices)
        previous_count = pole_counts[previous_side]
        previous_cluster = np.exp(4.0 * (np.sqrt(np.arange(1, previous_count + 1)) - np.sqrt(previous_count)))
        parameters = np.concatenate((parameters, length - previous_cluster[previous_cluster < length] / 3.0))
        parameters = np.unique(np.sort(parameters))
        direction = (end - start) / length
        points.extend(start + parameters[:, None] * direction)
        tangents.extend(np.repeat(direction[None, :], parameters.size, axis=0))
        sides.extend(np.full(parameters.size, side, dtype=int))
    points_array = np.asarray(points, dtype=float)
    return points_array[:, 0] + 1j * points_array[:, 1], np.asarray(tangents), np.asarray(sides)


def _arnoldi_basis(z: np.ndarray, center: complex, scale: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.ones(z.size, dtype=np.complex128)]
    hessenberg = np.zeros((degree + 1, degree), dtype=np.complex128)
    for column in range(degree):
        value = (z - center) * columns[column]
        for previous in range(column + 1):
            hessenberg[previous, column] = np.vdot(columns[previous], value) / z.size
            value -= hessenberg[previous, column] * columns[previous]
        denominator = np.linalg.norm(value) / np.sqrt(z.size)
        if denominator <= np.finfo(float).eps:
            return np.column_stack(columns), hessenberg
        hessenberg[column + 1, column] = denominator
        columns.append(value / denominator)
    return np.column_stack(columns), hessenberg


def _boundary_function_value(
    boundary_data: BoundaryData,
    z: np.ndarray,
    side_indices: np.ndarray,
    number_of_sides: int,
) -> np.ndarray:
    """Evaluate global or MATLAB-style per-side Dirichlet data."""
    if callable(boundary_data):
        values = boundary_data(z.real, z.imag)
        return np.asarray(values, dtype=float).reshape(z.shape)

    if len(boundary_data) != number_of_sides:
        raise ValueError(
            "per-side boundary data must contain exactly one value or function "
            f"for each of the {number_of_sides} polygon sides"
        )

    values = np.empty(z.shape, dtype=float)
    for side, side_data in enumerate(boundary_data):
        mask = side_indices == side
        if not np.any(mask):
            continue
        if callable(side_data):
            side_values = side_data(z[mask].real, z[mask].imag)
        else:
            side_values = side_data
        values[mask] = np.asarray(side_values, dtype=float)
    return values


def solve_laplace(
    vertices: np.ndarray,
    boundary_data: BoundaryData,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 30,
    max_poles_per_corner: int = 100,
    initial_degree_step: int = 4,
    use_arnoldi: bool = True,
    relative_corner_weight: bool = False,
    minimum_samples_per_side: int = 30,
    corner_mask: Optional[np.ndarray] = None,
) -> LightningLaplaceSolution:
    """Solve ``Delta u = 0`` with Dirichlet data on a polygon.

    ``vertices`` are Cartesian polygon corners.  They may be clockwise; they
    are normalized to counter-clockwise order internally.  ``boundary_data``
    may be one global function receiving vectorized ``(x, y)`` arrays, or a
    sequence with one scalar/function per side, matching MATLAB's ``g{k}``.
    Per-side functions also receive vectorized ``(x, y)`` arrays.

    ``corner_mask`` is an optional boolean array, one entry per vertex,
    marking which vertices are genuine corners eligible for pole placement
    (True) versus smooth points introduced only to approximate a curved
    boundary segment with straight lines (False, default True everywhere if
    omitted). The lightning method's poles exist to resolve the singular
    boundary-derivative behavior at real corners; placing them at a smooth
    point on a fine polyline approximation of a curve serves no purpose and
    can exhaust the solver's pole/matrix-size budget before it converges.
    Points marked False rely solely on the growing polynomial/Arnoldi basis
    to fit the (smooth) boundary data there.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1 or max_poles_per_corner < 1:
        raise ValueError("max_iterations and max_poles_per_corner must be positive")

    polygon, was_reversed = _as_ccw_vertices(vertices)
    if corner_mask is not None:
        corner_mask = np.asarray(corner_mask, dtype=bool)
        if corner_mask.shape[0] != polygon.shape[0]:
            raise ValueError("corner_mask must have one entry per polygon vertex")
        if was_reversed:
            corner_mask = corner_mask[::-1].copy()
    corner_z, side_lengths, outward, scale, center = _polygon_geometry(polygon)
    pole_counts = np.zeros(polygon.shape[0], dtype=int)
    best = None
    previous_error = np.inf

    for iteration in range(1, max_iterations + 1):
        poles = []
        pole_scales = []
        for corner, direction, count in zip(corner_z, outward, pole_counts):
            distances = scale * np.exp(4.0 * (np.sqrt(np.arange(1, count + 1)) - np.sqrt(count)))
            distances = distances[distances > 1e-15 * scale]
            candidate = corner + direction * distances
            keep = ~_inside_polygon(np.column_stack((candidate.real, candidate.imag)), polygon)
            poles.extend(candidate[keep])
            pole_scales.extend(distances[keep])

        poles_array = np.asarray(poles, dtype=np.complex128)
        pole_scales_array = np.asarray(pole_scales, dtype=float)
        samples, _, sample_sides = _boundary_samples(polygon, side_lengths, pole_counts, minimum_samples_per_side)
        values = _boundary_function_value(
            boundary_data,
            samples,
            sample_sides,
            polygon.shape[0],
        )
        sample_weights = np.ones(samples.size, dtype=float)
        if relative_corner_weight:
            distances_to_corners = np.min(np.abs(samples[:, None] - corner_z[None, :]), axis=1)
            sample_weights = distances_to_corners / scale

        degree = initial_degree_step * iteration
        if use_arnoldi:
            polynomial_basis, hessenberg = _arnoldi_basis(samples, center, scale, degree)
        else:
            hessenberg = None
            polynomial_basis = ((samples - center) / scale)[:, None] ** np.arange(degree + 1)

        matrix = np.column_stack((polynomial_basis.real, polynomial_basis[:, 1:].imag))
        if poles_array.size:
            matrix = np.column_stack((matrix, (pole_scales_array[None, :] / (samples[:, None] - poles_array[None, :])).real, (pole_scales_array[None, :] / (samples[:, None] - poles_array[None, :])).imag))

        weighted_matrix = matrix * np.sqrt(sample_weights)[:, None]
        weighted_values = values * np.sqrt(sample_weights)
        coefficients, _, _, _ = lstsq(weighted_matrix, weighted_values, lapack_driver="gelsd")
        residual = np.abs(matrix @ coefficients - values)
        corner_errors = np.array([np.max(residual[sample_sides == side]) for side in range(polygon.shape[0])])
        error = float(np.max(sample_weights * residual))

        if error < previous_error:
            polynomial_count = 2 * degree + 1
            best = (coefficients.copy(), poles_array.copy(), pole_scales_array.copy(), hessenberg, degree, error, samples.copy(), values.copy())
            previous_error = error

        if error < 0.5 * tolerance:
            break
        for side, side_error in enumerate(corner_errors):
            if corner_mask is not None and not corner_mask[side]:
                continue  # smooth boundary point: no pole singularity to resolve
            if side_error > 0.5 * error and pole_counts[side] < max_poles_per_corner:
                pole_counts[side] += int(np.ceil(1.0 + np.sqrt(pole_counts[side])))
            else:
                pole_counts[side] = max(pole_counts[side], int(np.ceil(iteration / 2)))

        if matrix.shape[1] > 1200 or np.sum(pole_counts) >= max_poles_per_corner * polygon.shape[0]:
            break

    if best is None:
        raise RuntimeError("Lightning solver did not produce a least-squares solution")

    coefficients, poles, pole_scales, hessenberg, degree, error, samples, values = best
    polynomial_coefficients = np.concatenate(([coefficients[0]], coefficients[1:degree + 1] - 1j * coefficients[degree + 1:2 * degree + 1]))
    pole_offset = 2 * degree + 1
    pole_coefficients = (
        coefficients[pole_offset:pole_offset + poles.size]
        - 1j * coefficients[pole_offset + poles.size:]
    )
    return LightningLaplaceSolution(
        vertices=polygon,
        center=center,
        scale=scale,
        polynomial_coefficients=polynomial_coefficients,
        pole_coefficients=pole_coefficients,
        poles=poles,
        pole_scales=pole_scales,
        arnoldi_hessenberg=hessenberg,
        degree=degree,
        max_error=error,
        iterations=iteration,
        boundary_points=samples,
        boundary_values=values,
    )


_SOLUTION_CACHE: dict = {}


def clear_solution_cache() -> None:
    """Drop all cached solve_case_11/solve_case_12 solutions.

    Call this if PDE_testcases.CASE_POLYGON_GEOMETRY is changed at runtime
    (e.g. interactively in a notebook) and a fresh solve is needed.
    """
    _SOLUTION_CACHE.clear()


def solve_case_11(*, use_cache: bool = True, **solver_options) -> LightningLaplaceSolution:
    """Solve the harmonic lifting problem for ``PDE_testcases.FUNCTION_CASE == 11``.

    Cached by default (keyed on ``solver_options``): reconstructing the
    gradient and Laplacian of this lifting
    (``inhomogenous_boundary._reconstruct_gradient_and_laplacian``)
    re-evaluates the same solution used for the plain value, so caching
    avoids repeating the iterative least-squares solve for every one of
    those calls. Pass ``use_cache=False`` to force a fresh solve.
    """
    from Geomertry import regular_polygon_vertices_np
    import PDE_testcases

    if PDE_testcases.FUNCTION_CASE != 11:
        raise ValueError("solve_case_11 requires PDE_testcases.FUNCTION_CASE == 11")

    key = ("case11", tuple(sorted(solver_options.items())))
    if use_cache and key in _SOLUTION_CACHE:
        return _SOLUTION_CACHE[key]

    side_data = [
        lambda x, y: 0 * x,
        lambda x, y: 1 - (x**2 + y**2),
        lambda x, y: 1 - x**2 - y**2,
        lambda x, y: 1 - (x**2 + y**2),
        lambda x, y: 0 * x
    ]
    vertices = regular_polygon_vertices_np(n_sides=5, radius=1.0, center=(0.0, 0.0))
    solution = solve_laplace(vertices, side_data, **solver_options)
    if use_cache:
        _SOLUTION_CACHE[key] = solution
    return solution


def solve_case_12(samples_per_side: int = 64, *, use_cache: bool = True, **solver_options) -> LightningLaplaceSolution:
    """Solve the harmonic lifting problem for ``PDE_testcases.FUNCTION_CASE == 12``
    (the mixed pentagon: 3 curved sides, 2 straight sides).

    The lightning method needs a polygon with straight edges, so each curved
    side is approximated by a fine polyline of ``samples_per_side`` straight
    segments sampled along the exact cubic Bezier-equivalent curve used in
    ``inhomogenous_boundary._build_polygon_boundary_data``; straight sides
    stay a single segment. Geometry comes from
    ``PDE_testcases.CASE_POLYGON_GEOMETRY[12]``, so it always matches the
    domain used elsewhere for case 12.

    Boundary data follows ``create_gt_fun4_mixed.m``: u = 0 on the straight
    sides, u = radius^2 - (x^2 + y^2) on the curved sides (matching
    ``PDE_testcases.dirichletBoundary_vectorized``'s FUNCTION_CASE==12
    branch, which uses radius^2 rather than 1^2 so this vanishes exactly at
    the pentagon's own vertices).

    Cached by default, like ``solve_case_11`` -- see its docstring. Pass
    ``use_cache=False`` to force a fresh solve.
    """
    import PDE_testcases

    if PDE_testcases.FUNCTION_CASE != 12:
        raise ValueError("solve_case_12 requires PDE_testcases.FUNCTION_CASE == 12")
    if samples_per_side < 1:
        raise ValueError("samples_per_side must be positive")

    params = PDE_testcases.get_case_polygon_geometry(12)
    radius = float(params["radius"])
    center = params["center"]
    rotation = float(params["rotation"])
    bulge = float(params["bulge"])
    curved_side_indices = set(int(i) for i in params["curved_side_indices"])

    key = (
        "case12", samples_per_side, radius, tuple(center), rotation, bulge,
        tuple(sorted(curved_side_indices)), tuple(sorted(solver_options.items())),
    )
    if use_cache and key in _SOLUTION_CACHE:
        return _SOLUTION_CACHE[key]

    n_sides = 5
    angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=float)
    corners = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])

    zero_fn = lambda x, y: 0.0 * x
    # radius**2, not 1**2: this must vanish at the pentagon's own vertices
    # (|z|=radius), matching PDE_testcases.dirichletBoundary_vectorized's
    # FUNCTION_CASE==12 branch -- see solve_case_12's docstring.
    circle_fn = lambda x, y: radius**2 - (x**2 + y**2)

    fine_vertices = []
    side_data: list = []
    corner_mask: list = []
    for side_idx in range(n_sides):
        v0, v1 = corners[side_idx], corners[(side_idx + 1) % n_sides]

        if side_idx in curved_side_indices:
            # Same cubic Bezier-equivalent span as
            # inhomogenous_boundary._build_polygon_boundary_data: an inward
            # bulge of `bulge * edge_len` at the two interior control points.
            edge = v1 - v0
            edge_len = float(np.linalg.norm(edge))
            n_in = np.array([-edge[1], edge[0]]) / edge_len
            curv = bulge * edge_len
            p1 = (2.0 / 3.0) * v0 + (1.0 / 3.0) * v1 + curv * n_in
            p2 = (1.0 / 3.0) * v0 + (2.0 / 3.0) * v1 + curv * n_in

            t = np.linspace(0.0, 1.0, samples_per_side, endpoint=False)[:, None]
            u = 1.0 - t
            points = (u**3) * v0 + (3 * u**2 * t) * p1 + (3 * u * t**2) * p2 + (t**3) * v1
            fine_vertices.append(points)
            side_data.extend([circle_fn] * samples_per_side)
            # Only the first point (t=0) is a real pentagon corner; the rest
            # are smooth interior points of the curve, not corners (see
            # solve_laplace's corner_mask).
            corner_mask.extend([True] + [False] * (samples_per_side - 1))
        else:
            fine_vertices.append(v0[None, :])
            side_data.append(zero_fn)
            corner_mask.append(True)

    vertices = np.vstack(fine_vertices)
    solution = solve_laplace(vertices, side_data, corner_mask=np.array(corner_mask, dtype=bool), **solver_options)
    if use_cache:
        _SOLUTION_CACHE[key] = solution
    return solution
