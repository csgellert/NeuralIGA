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

    def evaluate(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """Evaluate the real harmonic solution at complex or Cartesian points."""
        if y is None:
            z = np.asarray(x, dtype=np.complex128)
        else:
            x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            z = x_arr + 1j * y_arr
        return np.real(self._evaluate_complex(z))

    def __call__(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        return self.evaluate(x, y)


def _as_ccw_vertices(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 3:
        raise ValueError("vertices must have shape (number_of_vertices, 2)")
    area2 = np.sum(values[:, 0] * np.roll(values[:, 1], -1) - values[:, 1] * np.roll(values[:, 0], -1))
    return values if area2 >= 0 else values[::-1].copy()


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


def _inside_convex_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = points[:, None, :] - vertices[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-13, axis=1)


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
) -> LightningLaplaceSolution:
    """Solve ``Delta u = 0`` with Dirichlet data on a polygon.

    ``vertices`` are Cartesian polygon corners.  They may be clockwise; they
    are normalized to counter-clockwise order internally.  ``boundary_data``
    may be one global function receiving vectorized ``(x, y)`` arrays, or a
    sequence with one scalar/function per side, matching MATLAB's ``g{k}``.
    Per-side functions also receive vectorized ``(x, y)`` arrays.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1 or max_poles_per_corner < 1:
        raise ValueError("max_iterations and max_poles_per_corner must be positive")

    polygon = _as_ccw_vertices(vertices)
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
            keep = ~_inside_convex_polygon(np.column_stack((candidate.real, candidate.imag)), polygon)
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


def solve_case_11(**solver_options) -> LightningLaplaceSolution:
    """Solve the harmonic lifting problem for ``PDE_testcases.FUNCTION_CASE == 11``."""
    from Geomertry import regular_polygon_vertices_np
    import PDE_testcases

    if PDE_testcases.FUNCTION_CASE != 11:
        raise ValueError("solve_case_11 requires PDE_testcases.FUNCTION_CASE == 11")
    side_data = [
        lambda x, y: 0 * x,
        lambda x, y: 1 - (x**2 + y**2),
        lambda x, y: 1 - x**2 - y**2,
        lambda x, y: 1 - (x**2 + y**2),
        lambda x, y: 0 * x 
    ]
    vertices = regular_polygon_vertices_np(n_sides=5, radius=1.0, center=(0.0, 0.0))
    return solve_laplace(vertices, side_data, **solver_options)
