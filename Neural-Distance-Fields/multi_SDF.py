import numpy as np
import matplotlib.pyplot as plt
import torch


_CURVED_PENTAGON_CACHE = {}
_MIXED_PENTAGON_CACHE = {}

def _as_torch_tensor(value, *, device, dtype):
	if torch.is_tensor(value):
		return value.to(device=device, dtype=dtype)
	return torch.as_tensor(value, device=device, dtype=dtype)


def _curved_pentagon_cache_key(radius, center, rotation, bulge, samples_per_side):
	return (
		float(radius),
		(float(center[0]), float(center[1])),
		float(rotation),
		float(bulge),
		int(samples_per_side),
	)


def _mixed_pentagon_cache_key(radius, center, rotation, bulge, samples_per_side, curved_side_indices):
	return (
		float(radius),
		(float(center[0]), float(center[1])),
		float(rotation),
		float(bulge),
		int(samples_per_side),
		tuple(int(i) for i in curved_side_indices),
	)


def _cubic_bezier_eval(p0, p1, p2, p3, t):
	"""Evaluate cubic B-spline span in clamped form (Bezier-equivalent)."""
	one_minus_t = 1.0 - t
	b0 = one_minus_t ** 3
	b1 = 3.0 * one_minus_t ** 2 * t
	b2 = 3.0 * one_minus_t * t ** 2
	b3 = t ** 3
	return (
		b0[:, None] * p0[None, :]
		+ b1[:, None] * p1[None, :]
		+ b2[:, None] * p2[None, :]
		+ b3[:, None] * p3[None, :]
	)


def _build_curved_pentagon_bspline_cache(
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
):
	"""
	Build and cache geometry for a pentagon-like closed curve made of 5 cubic B-spline spans.

	Each side is represented by one clamped cubic span (Bezier-equivalent control layout),
	which gives smooth, curved sides while preserving a pentagon-like topology.
	"""
	if samples_per_side < 8:
		raise ValueError("samples_per_side must be >= 8")
	if radius <= 0.0:
		raise ValueError("radius must be positive")

	key = _curved_pentagon_cache_key(radius, center, rotation, bulge, samples_per_side)
	if key in _CURVED_PENTAGON_CACHE:
		return _CURVED_PENTAGON_CACHE[key]

	cx, cy = float(center[0]), float(center[1])
	n_sides = 5
	angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)
	vertices = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

	t_vals = np.linspace(0.0, 1.0, int(samples_per_side) + 1, dtype=np.float64)
	side_polylines = []
	for i in range(n_sides):
		v0 = vertices[i]
		v1 = vertices[(i + 1) % n_sides]
		edge = v1 - v0
		edge_len = np.linalg.norm(edge)
		if edge_len <= 1e-14:
			raise ValueError("Degenerate pentagon edge encountered")

		# Inward normal for CCW-ordered vertices.
		n_in = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_len
		curv = float(bulge) * edge_len

		p0 = v0
		p1 = (2.0 / 3.0) * v0 + (1.0 / 3.0) * v1 + curv * n_in
		p2 = (1.0 / 3.0) * v0 + (2.0 / 3.0) * v1 + curv * n_in
		p3 = v1

		side_curve = _cubic_bezier_eval(p0, p1, p2, p3, t_vals)
		side_polylines.append(side_curve)

	boundary_parts = [side_polylines[i][:-1] for i in range(n_sides)]
	boundary_polyline = np.vstack(boundary_parts)
	x_min = float(np.min(boundary_polyline[:, 0]))
	x_max = float(np.max(boundary_polyline[:, 0]))
	y_min = float(np.min(boundary_polyline[:, 1]))
	y_max = float(np.max(boundary_polyline[:, 1]))

	cache = {
		"n_sides": n_sides,
		"side_polylines": side_polylines,
		"boundary_polyline": boundary_polyline,
		"bbox": (x_min, x_max, y_min, y_max),
	}
	_CURVED_PENTAGON_CACHE[key] = cache
	return cache


def _build_mixed_pentagon_bspline_cache(
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	curved_side_indices=(0, 2, 4),
):
	"""
	Build and cache a mixed pentagon boundary with 3 curved and 2 straight sides.

	Curved sides use one cubic Bezier span per side, while straight sides are
	kept as line segments sampled into polylines.
	"""
	if samples_per_side < 8:
		raise ValueError("samples_per_side must be >= 8")
	if radius <= 0.0:
		raise ValueError("radius must be positive")

	curved_side_indices = tuple(sorted(set(int(i) for i in curved_side_indices)))
	if len(curved_side_indices) != 3:
		raise ValueError("curved_side_indices must contain exactly 3 unique side indices")
	if any((i < 0 or i >= 5) for i in curved_side_indices):
		raise ValueError("curved_side_indices entries must be in [0, 4]")

	key = _mixed_pentagon_cache_key(radius, center, rotation, bulge, samples_per_side, curved_side_indices)
	if key in _MIXED_PENTAGON_CACHE:
		return _MIXED_PENTAGON_CACHE[key]

	cx, cy = float(center[0]), float(center[1])
	n_sides = 5
	angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)
	vertices = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

	t_vals = np.linspace(0.0, 1.0, int(samples_per_side) + 1, dtype=np.float64)
	curved_set = set(curved_side_indices)
	side_polylines = []
	for i in range(n_sides):
		v0 = vertices[i]
		v1 = vertices[(i + 1) % n_sides]
		edge = v1 - v0
		edge_len = np.linalg.norm(edge)
		if edge_len <= 1e-14:
			raise ValueError("Degenerate pentagon edge encountered")

		if i in curved_set:
			# Inward normal for CCW-ordered vertices.
			n_in = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_len
			curv = float(bulge) * edge_len

			p0 = v0
			p1 = (2.0 / 3.0) * v0 + (1.0 / 3.0) * v1 + curv * n_in
			p2 = (1.0 / 3.0) * v0 + (2.0 / 3.0) * v1 + curv * n_in
			p3 = v1
			side_curve = _cubic_bezier_eval(p0, p1, p2, p3, t_vals)
		else:
			side_curve = v0[None, :] + t_vals[:, None] * edge[None, :]

		side_polylines.append(side_curve)

	boundary_parts = [side_polylines[i][:-1] for i in range(n_sides)]
	boundary_polyline = np.vstack(boundary_parts)
	x_min = float(np.min(boundary_polyline[:, 0]))
	x_max = float(np.max(boundary_polyline[:, 0]))
	y_min = float(np.min(boundary_polyline[:, 1]))
	y_max = float(np.max(boundary_polyline[:, 1]))

	cache = {
		"n_sides": n_sides,
		"side_polylines": side_polylines,
		"boundary_polyline": boundary_polyline,
		"bbox": (x_min, x_max, y_min, y_max),
		"curved_side_indices": curved_side_indices,
	}
	_MIXED_PENTAGON_CACHE[key] = cache
	return cache


def _point_to_polyline_distance_numpy(points, polyline):
	"""Minimum distance from points (P,2) to polyline segments."""
	start = polyline[:-1]
	end = polyline[1:]
	edges = end - start
	edge_len_sq = np.einsum("ni,ni->n", edges, edges, optimize=True)
	edge_len_sq = np.maximum(edge_len_sq, 1e-14)

	deltas = points[:, None, :] - start[None, :, :]
	t = np.einsum("pni,ni->pn", deltas, edges, optimize=True) / edge_len_sq[None, :]
	t = np.clip(t, 0.0, 1.0)
	closest = start[None, :, :] + t[:, :, None] * edges[None, :, :]
	d = np.linalg.norm(points[:, None, :] - closest, axis=2)
	return np.min(d, axis=1)


def _point_to_polyline_distance_torch(points, polyline_t):
	"""Minimum distance from points (P,2) to polyline segments in torch."""
	start = polyline_t[:-1]
	end = polyline_t[1:]
	edges = end - start
	edge_len_sq = torch.einsum("ni,ni->n", edges, edges)
	edge_len_sq = torch.clamp(edge_len_sq, min=1e-14)

	deltas = points[:, None, :] - start[None, :, :]
	t = torch.einsum("pni,ni->pn", deltas, edges) / edge_len_sq[None, :]
	t = torch.clamp(t, 0.0, 1.0)
	closest = start[None, :, :] + t[:, :, None] * edges[None, :, :]
	d = torch.linalg.norm(points[:, None, :] - closest, dim=2)
	return torch.min(d, dim=1).values


def _points_in_polygon_numpy(points, polygon):
	"""Vectorized odd-even rule test for points in simple polygon."""
	x = points[:, 0]
	y = points[:, 1]
	xi = polygon[:, 0]
	yi = polygon[:, 1]
	xj = np.roll(xi, -1)
	yj = np.roll(yi, -1)
	eps = 1e-14
	intersects = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])) & (
		x[:, None] < (xj[None, :] - xi[None, :]) * (y[:, None] - yi[None, :]) / (yj[None, :] - yi[None, :] + eps) + xi[None, :]
	)
	return (np.count_nonzero(intersects, axis=1) % 2) == 1


def _points_in_polygon_torch(points, polygon_t):
	"""Vectorized odd-even rule test in torch."""
	x = points[:, 0]
	y = points[:, 1]
	xi = polygon_t[:, 0]
	yi = polygon_t[:, 1]
	xj = torch.roll(xi, shifts=-1, dims=0)
	yj = torch.roll(yi, shifts=-1, dims=0)
	eps = torch.as_tensor(1e-14, device=points.device, dtype=points.dtype)
	intersects = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])) & (
		x[:, None] < (xj[None, :] - xi[None, :]) * (y[:, None] - yi[None, :]) / (yj[None, :] - yi[None, :] + eps) + xi[None, :]
	)
	return (torch.count_nonzero(intersects, dim=1) % 2) == 1


def is_inside_curved_pentagon_bspline(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	return_numpy=False,
):
	"""Point-in-domain test for the closed curved pentagon B-spline-like boundary."""
	cache = _build_curved_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
	)
	polygon = cache["boundary_polyline"]

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)
		poly_t = torch.as_tensor(polygon, device=device, dtype=dtype)
		inside = _points_in_polygon_torch(pts, poly_t).reshape(x_arr.shape)
		if return_numpy:
			return inside.detach().cpu().numpy()
		return inside

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")
	pts = np.column_stack((x_arr.reshape(-1), y_arr.reshape(-1)))
	inside = _points_in_polygon_numpy(pts, polygon).reshape(x_arr.shape)
	if return_numpy:
		return inside
	return inside.tolist()


def is_inside_mixed_pentagon_bspline(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	curved_side_indices=(0, 2, 4),
	return_numpy=False,
):
	"""Point-in-domain test for mixed pentagon with 3 curved and 2 straight sides."""
	cache = _build_mixed_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
		curved_side_indices=curved_side_indices,
	)
	polygon = cache["boundary_polyline"]

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)
		poly_t = torch.as_tensor(polygon, device=device, dtype=dtype)
		inside = _points_in_polygon_torch(pts, poly_t).reshape(x_arr.shape)
		if return_numpy:
			return inside.detach().cpu().numpy()
		return inside

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")
	pts = np.column_stack((x_arr.reshape(-1), y_arr.reshape(-1)))
	inside = _points_in_polygon_numpy(pts, polygon).reshape(x_arr.shape)
	if return_numpy:
		return inside
	return inside.tolist()


def curved_pentagon_bspline_side_distances(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	use_sign=False,
	return_numpy=False,
):
	"""
	Per-side distances to a pentagon-like curved boundary defined by cubic B-spline spans.

	Returns shape (P, 5) for P points or (5,) for scalar inputs.
	"""
	cache = _build_curved_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
	)
	side_polylines = cache["side_polylines"]

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		scalar_input = x_arr.ndim == 0
		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)
		dists = []
		for poly in side_polylines:
			poly_t = torch.as_tensor(poly, device=device, dtype=dtype)
			dists.append(_point_to_polyline_distance_torch(pts, poly_t))
		distances = torch.stack(dists, dim=1)

		if use_sign:
			inside = is_inside_curved_pentagon_bspline(
				pts[:, 0],
				pts[:, 1],
				radius=radius,
				center=center,
				rotation=rotation,
				bulge=bulge,
				samples_per_side=samples_per_side,
			)
			sign = torch.where(inside, torch.ones_like(pts[:, 0]), -torch.ones_like(pts[:, 0]))
			distances = distances * sign[:, None]

		if scalar_input:
			distances = distances[0]
		if return_numpy:
			return distances.detach().cpu().numpy()
		return distances

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	scalar_input = x_arr.ndim == 0
	pts = np.column_stack((x_arr.reshape(-1), y_arr.reshape(-1)))
	distances = np.column_stack([_point_to_polyline_distance_numpy(pts, poly) for poly in side_polylines])

	if use_sign:
		inside = is_inside_curved_pentagon_bspline(
			pts[:, 0],
			pts[:, 1],
			radius=radius,
			center=center,
			rotation=rotation,
			bulge=bulge,
			samples_per_side=samples_per_side,
			return_numpy=True,
		)
		sign = np.where(inside, 1.0, -1.0)
		distances = distances * sign[:, None]

	if scalar_input:
		distances = distances[0]
	if return_numpy:
		return distances
	return distances.tolist()


def mixed_pentagon_bspline_side_distances(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	curved_side_indices=(0, 2, 4),
	use_sign=False,
	return_numpy=False,
):
	"""
	Per-side distances to a mixed pentagon boundary with 3 curved and 2 straight sides.

	Returns shape (P, 5) for P points or (5,) for scalar inputs.
	"""
	cache = _build_mixed_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
		curved_side_indices=curved_side_indices,
	)
	side_polylines = cache["side_polylines"]

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		scalar_input = x_arr.ndim == 0
		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)
		dists = []
		for poly in side_polylines:
			poly_t = torch.as_tensor(poly, device=device, dtype=dtype)
			dists.append(_point_to_polyline_distance_torch(pts, poly_t))
		distances = torch.stack(dists, dim=1)

		if use_sign:
			inside = is_inside_mixed_pentagon_bspline(
				pts[:, 0],
				pts[:, 1],
				radius=radius,
				center=center,
				rotation=rotation,
				bulge=bulge,
				samples_per_side=samples_per_side,
				curved_side_indices=curved_side_indices,
			)
			sign = torch.where(inside, torch.ones_like(pts[:, 0]), -torch.ones_like(pts[:, 0]))
			distances = distances * sign[:, None]

		if scalar_input:
			distances = distances[0]
		if return_numpy:
			return distances.detach().cpu().numpy()
		return distances

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	scalar_input = x_arr.ndim == 0
	pts = np.column_stack((x_arr.reshape(-1), y_arr.reshape(-1)))
	distances = np.column_stack([_point_to_polyline_distance_numpy(pts, poly) for poly in side_polylines])

	if use_sign:
		inside = is_inside_mixed_pentagon_bspline(
			pts[:, 0],
			pts[:, 1],
			radius=radius,
			center=center,
			rotation=rotation,
			bulge=bulge,
			samples_per_side=samples_per_side,
			curved_side_indices=curved_side_indices,
			return_numpy=True,
		)
		sign = np.where(inside, 1.0, -1.0)
		distances = distances * sign[:, None]

	if scalar_input:
		distances = distances[0]
	if return_numpy:
		return distances
	return distances.tolist()


def generate_curved_pentagon_bspline_boundary_points(
	num_points,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	device=None,
	return_side_indices=False,
):
	"""Sample points along curved B-spline pentagon boundary, optionally returning side index."""
	cache = _build_curved_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
	)
	side_polylines = cache["side_polylines"]
	n_sides = 5

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	side_idx = torch.randint(0, n_sides, (int(num_points),), device=device)
	t = torch.rand(int(num_points), device=device)
	pts = torch.empty(int(num_points), 2, device=device, dtype=torch.float32)

	for i in range(n_sides):
		mask = side_idx == i
		if not torch.any(mask):
			continue
		poly = torch.as_tensor(side_polylines[i], device=device, dtype=torch.float32)
		n_seg = poly.shape[0] - 1
		local_t = t[mask] * n_seg
		seg_idx = torch.clamp(torch.floor(local_t).long(), max=n_seg - 1)
		alpha = (local_t - seg_idx.to(local_t.dtype)).unsqueeze(1)
		p0 = poly[seg_idx]
		p1 = poly[seg_idx + 1]
		pts[mask] = (1.0 - alpha) * p0 + alpha * p1

	if return_side_indices:
		return pts, side_idx
	return pts


def generate_mixed_pentagon_bspline_boundary_points(
	num_points,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	bulge=0.18,
	samples_per_side=128,
	curved_side_indices=(0, 2, 4),
	device=None,
	return_side_indices=False,
):
	"""Sample points along mixed pentagon boundary, optionally returning side index."""
	cache = _build_mixed_pentagon_bspline_cache(
		radius=radius,
		center=center,
		rotation=rotation,
		bulge=bulge,
		samples_per_side=samples_per_side,
		curved_side_indices=curved_side_indices,
	)
	side_polylines = cache["side_polylines"]
	n_sides = 5

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	side_idx = torch.randint(0, n_sides, (int(num_points),), device=device)
	t = torch.rand(int(num_points), device=device)
	pts = torch.empty(int(num_points), 2, device=device, dtype=torch.float32)

	for i in range(n_sides):
		mask = side_idx == i
		if not torch.any(mask):
			continue
		poly = torch.as_tensor(side_polylines[i], device=device, dtype=torch.float32)
		n_seg = poly.shape[0] - 1
		local_t = t[mask] * n_seg
		seg_idx = torch.clamp(torch.floor(local_t).long(), max=n_seg - 1)
		alpha = (local_t - seg_idx.to(local_t.dtype)).unsqueeze(1)
		p0 = poly[seg_idx]
		p1 = poly[seg_idx + 1]
		pts[mask] = (1.0 - alpha) * p0 + alpha * p1

	if return_side_indices:
		return pts, side_idx
	return pts


def regular_ngon_side_signed_distances(
	x,
	y,
	n_sides,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	use_sign=False,
	return_numpy=False,
):
	"""
	Vectorized distance from each side segment of a regular n-gon.

	The polygon is centered at ``center`` and built counter-clockwise.
	By default, this function returns unsigned point-to-segment distances.
	Set ``use_sign=True`` to return signed distances using the convention:
	- positive: point is on the inner side of the edge
	- negative: point is on the outer side of the edge

	Args:
		x: scalar or array-like x coordinates.
		y: scalar or array-like y coordinates (same shape as x).
		n_sides: number of polygon sides (>= 3).
		radius: circumradius of the regular n-gon.
		center: tuple (cx, cy).
		rotation: angular offset in radians.
		use_sign: if True, apply inward-half-plane sign to distances.
		return_numpy: if True, return a NumPy array instead of Python list.

	Returns:
		If scalar input: shape (n_sides,) distances for that point.
		If array input with P points: shape (P, n_sides) distances.
		NumPy input returns NumPy output by default; torch input returns a torch tensor.
	"""
	if n_sides < 3:
		raise ValueError("n_sides must be at least 3")

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)

		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		scalar_input = x_arr.ndim == 0
		points = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)

		center_t = torch.as_tensor(center, device=device, dtype=dtype)
		rotation_t = torch.as_tensor(rotation, device=device, dtype=dtype)
		radius_t = torch.as_tensor(radius, device=device, dtype=dtype)
		angles = rotation_t + (2.0 * torch.pi / n_sides) * torch.arange(n_sides, device=device, dtype=dtype)

		vertices = torch.stack(
			(
				center_t[0] + radius_t * torch.cos(angles),
				center_t[1] + radius_t * torch.sin(angles),
			),
			dim=1,
		)
		v_next = torch.roll(vertices, shifts=-1, dims=0)
		edges = v_next - vertices
		normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
		normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)

		deltas = points[:, None, :] - vertices[None, :, :]
		edge_len_sq = torch.einsum("ni,ni->n", edges, edges)
		t = torch.einsum("pni,ni->pn", deltas, edges) / edge_len_sq[None, :]
		t = torch.clamp(t, 0.0, 1.0)
		closest = vertices[None, :, :] + t[:, :, None] * edges[None, :, :]
		seg_delta = points[:, None, :] - closest
		dist_mag = torch.linalg.norm(seg_delta, dim=2)

		if use_sign:
			line_signed = torch.einsum("pni,ni->pn", deltas, normals)
			sign = torch.where(line_signed >= 0.0, torch.ones_like(line_signed), -torch.ones_like(line_signed))
			distances = dist_mag * sign
		else:
			distances = dist_mag

		if scalar_input:
			distances = distances[0]

		if return_numpy:
			return distances.detach().cpu().numpy()
		return distances

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)

	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	scalar_input = x_arr.ndim == 0

	# Flatten points for a single broadcasted computation.
	points = np.column_stack((x_arr.ravel(), y_arr.ravel()))

	cx, cy = center
	angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)

	vx = cx + radius * np.cos(angles)
	vy = cy + radius * np.sin(angles)
	vertices = np.column_stack((vx, vy))

	v_next = np.roll(vertices, -1, axis=0)
	edges = v_next - vertices

	# Inward normals for CCW-ordered vertices.
	normals = np.column_stack((-edges[:, 1], edges[:, 0]))
	normals /= np.linalg.norm(normals, axis=1, keepdims=True)

	# Point-to-segment distance magnitude with endpoint clamping.
	# Shapes:
	# points[:, None, :] -> (P, 1, 2)
	# vertices[None, :, :] -> (1, N, 2)
	deltas = points[:, None, :] - vertices[None, :, :]
	edge_len_sq = np.einsum("ni,ni->n", edges, edges, optimize=True)
	t = np.einsum("pni,ni->pn", deltas, edges, optimize=True) / edge_len_sq[None, :]
	t = np.clip(t, 0.0, 1.0)
	closest = vertices[None, :, :] + t[:, :, None] * edges[None, :, :]
	seg_delta = points[:, None, :] - closest
	dist_mag = np.linalg.norm(seg_delta, axis=2)

	if use_sign:
		# Sign is based on the edge's inward half-plane.
		line_signed = np.einsum("pni,ni->pn", deltas, normals, optimize=True)
		sign = np.where(line_signed >= 0.0, 1.0, -1.0)
		distances = dist_mag * sign
	else:
		distances = dist_mag

	if scalar_input:
		distances = distances[0]

	if return_numpy:
		return distances
	return distances.tolist()

def generate_regular_ngon_boundary_points(num_points,n_sides, radius=1.0, center=(0.0, 0.0), rotation=0.0):
	"""
	Generate boundary points along the edges of a regular n-gon.

	The polygon is centered at ``center`` and built counter-clockwise.

	Args:
		n_sides: number of polygon sides (>= 3).
		radius: circumradius of the regular n-gon.
		center: tuple (cx, cy).
		rotation: angular offset in radians.
	"""
	if n_sides < 3:
		raise ValueError("n_sides must be at least 3")

	cx, cy = center
	angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)

	vx = cx + radius * np.cos(angles)
	vy = cy + radius * np.sin(angles)
	vertices = np.column_stack((vx, vy))
	# choose a number of random points along the edges
	edge_points = []
	for i in range(n_sides):
		start = vertices[i]
		end = vertices[(i + 1) % n_sides]
		num_points = max(2, int(np.linalg.norm(end - start) * 10))  # 10 points per unit length
		t_values = np.linspace(0, 1, num_points, endpoint=False)
		points_on_edge = start + t_values[:, None] * (end - start)
		edge_points.append(points_on_edge)
	return np.vstack(edge_points)


def _point_to_segment_distance_numpy(points, start, end):
	"""Vectorized point-to-segment distance in NumPy.

	Args:
		points: (P, 2)
		start: (2,)
		end: (2,)
	Returns:
		(P,) distances
	"""
	edge = end - start
	edge_len_sq = np.dot(edge, edge)
	if edge_len_sq <= 0.0:
		return np.linalg.norm(points - start[None, :], axis=1)

	delta = points - start[None, :]
	t = np.einsum("pi,i->p", delta, edge, optimize=True) / edge_len_sq
	t = np.clip(t, 0.0, 1.0)
	closest = start[None, :] + t[:, None] * edge[None, :]
	return np.linalg.norm(points - closest, axis=1)


def _point_to_segment_distance_torch(points, start, end):
	"""Vectorized point-to-segment distance in PyTorch.

	Args:
		points: (P, 2)
		start: (2,)
		end: (2,)
	Returns:
		(P,) distances
	"""
	edge = end - start
	edge_len_sq = torch.dot(edge, edge)
	if float(edge_len_sq.item()) <= 0.0:
		return torch.linalg.norm(points - start.unsqueeze(0), dim=1)

	delta = points - start.unsqueeze(0)
	t = torch.einsum("pi,i->p", delta, edge) / edge_len_sq
	t = torch.clamp(t, 0.0, 1.0)
	closest = start.unsqueeze(0) + t.unsqueeze(1) * edge.unsqueeze(0)
	return torch.linalg.norm(points - closest, dim=1)


def _inside_triangle_numpy(points, a, b, c):
	"""Return boolean mask for points inside or on triangle (a, b, c)."""
	v0 = c - a
	v1 = b - a
	v2 = points - a[None, :]
	den = v0[0] * v1[1] - v1[0] * v0[1]
	if np.abs(den) < 1e-14:
		return np.zeros(points.shape[0], dtype=bool)
	inv_den = 1.0 / den
	u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) * inv_den
	v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) * inv_den
	w = 1.0 - u - v
	eps = 1e-12
	return (u >= -eps) & (v >= -eps) & (w >= -eps)


def _inside_triangle_torch(points, a, b, c):
	"""Return boolean mask for points inside or on triangle (a, b, c)."""
	v0 = c - a
	v1 = b - a
	v2 = points - a.unsqueeze(0)
	den = v0[0] * v1[1] - v1[0] * v0[1]
	if torch.abs(den) < 1e-14:
		return torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
	inv_den = 1.0 / den
	u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) * inv_den
	v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) * inv_den
	w = 1.0 - u - v
	eps = 1e-12
	return (u >= -eps) & (v >= -eps) & (w >= -eps)


def is_inside_semicircle_triangle_union(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	apex=(0.0, -1.0),
	return_numpy=False,
):
	"""
	Point-in-domain test for test-case-2 geometry: semicircle ∪ isosceles triangle.

	Geometry definition:
	- Semicircle: upper half of circle with given ``center`` and ``radius``.
	- Triangle: vertices A=(-r,0)+center, B=(+r,0)+center, and ``apex``.
	- Domain: union of the two sets.
	"""
	if radius <= 0.0:
		raise ValueError("radius must be positive")

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)
		cx, cy = center
		a = torch.as_tensor([cx - radius, cy], device=device, dtype=dtype)
		b = torch.as_tensor([cx + radius, cy], device=device, dtype=dtype)
		c = torch.as_tensor(apex, device=device, dtype=dtype)

		rel = pts - torch.as_tensor([cx, cy], device=device, dtype=dtype).unsqueeze(0)
		inside_semicircle = (rel[:, 0] ** 2 + rel[:, 1] ** 2 <= radius ** 2 + 1e-12) & (pts[:, 1] >= cy - 1e-12)
		inside_triangle = _inside_triangle_torch(pts, a, b, c)
		inside = inside_semicircle | inside_triangle

		inside = inside.reshape(x_arr.shape)
		if return_numpy:
			return inside.detach().cpu().numpy()
		return inside

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
	cx, cy = center
	a = np.asarray([cx - radius, cy], dtype=np.float64)
	b = np.asarray([cx + radius, cy], dtype=np.float64)
	c = np.asarray(apex, dtype=np.float64)

	rel = pts - np.asarray([cx, cy], dtype=np.float64)[None, :]
	inside_semicircle = (rel[:, 0] ** 2 + rel[:, 1] ** 2 <= radius ** 2 + 1e-12) & (pts[:, 1] >= cy - 1e-12)
	inside_triangle = _inside_triangle_numpy(pts, a, b, c)
	inside = (inside_semicircle | inside_triangle).reshape(x_arr.shape)

	if return_numpy:
		return inside
	return inside.tolist()


def semicircle_triangle_side_distances(
	x,
	y,
	radius=1.0,
	center=(0.0, 0.0),
	apex=(0.0, -1.0),
	return_numpy=False,
):
	"""
	Distance fields for test-case-2 boundary pieces (3 outputs):
	0: left line segment A->apex
	1: right line segment apex->B
	2: upper semicircle arc from A to B

	Returns shape (P, 3) for P query points, or (3,) for scalar input.
	"""
	if radius <= 0.0:
		raise ValueError("radius must be positive")

	if torch.is_tensor(x) or torch.is_tensor(y):
		device = x.device if torch.is_tensor(x) else y.device
		x_dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float64
		y_dtype = y.dtype if torch.is_tensor(y) and y.is_floating_point() else torch.float64
		dtype = torch.promote_types(x_dtype, y_dtype)

		x_arr = _as_torch_tensor(x, device=device, dtype=dtype)
		y_arr = _as_torch_tensor(y, device=device, dtype=dtype)
		if x_arr.shape != y_arr.shape:
			raise ValueError("x and y must have the same shape")

		scalar_input = x_arr.ndim == 0
		pts = torch.stack((x_arr.reshape(-1), y_arr.reshape(-1)), dim=1)

		cx, cy = center
		a = torch.as_tensor([cx - radius, cy], device=device, dtype=dtype)
		b = torch.as_tensor([cx + radius, cy], device=device, dtype=dtype)
		c = torch.as_tensor(apex, device=device, dtype=dtype)
		center_t = torch.as_tensor([cx, cy], device=device, dtype=dtype)

		d_left = _point_to_segment_distance_torch(pts, a, c)
		d_right = _point_to_segment_distance_torch(pts, c, b)

		rel = pts - center_t.unsqueeze(0)
		rho = torch.linalg.norm(rel, dim=1)
		theta = torch.atan2(rel[:, 1], rel[:, 0])
		on_arc = (theta >= 0.0) & (theta <= torch.pi)
		d_to_circle = torch.abs(rho - radius)
		d_to_a = torch.linalg.norm(pts - a.unsqueeze(0), dim=1)
		d_to_b = torch.linalg.norm(pts - b.unsqueeze(0), dim=1)
		d_endpoints = torch.minimum(d_to_a, d_to_b)
		d_arc = torch.where(on_arc, d_to_circle, d_endpoints)

		distances = torch.stack((d_left, d_right, d_arc), dim=1)
		if scalar_input:
			distances = distances[0]

		if return_numpy:
			return distances.detach().cpu().numpy()
		return distances

	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	if x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	scalar_input = x_arr.ndim == 0
	pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))

	cx, cy = center
	a = np.asarray([cx - radius, cy], dtype=np.float64)
	b = np.asarray([cx + radius, cy], dtype=np.float64)
	c = np.asarray(apex, dtype=np.float64)

	d_left = _point_to_segment_distance_numpy(pts, a, c)
	d_right = _point_to_segment_distance_numpy(pts, c, b)

	rel = pts - np.asarray([cx, cy], dtype=np.float64)[None, :]
	rho = np.linalg.norm(rel, axis=1)
	theta = np.arctan2(rel[:, 1], rel[:, 0])
	on_arc = (theta >= 0.0) & (theta <= np.pi)
	d_to_circle = np.abs(rho - radius)
	d_to_a = np.linalg.norm(pts - a[None, :], axis=1)
	d_to_b = np.linalg.norm(pts - b[None, :], axis=1)
	d_endpoints = np.minimum(d_to_a, d_to_b)
	d_arc = np.where(on_arc, d_to_circle, d_endpoints)

	distances = np.column_stack((d_left, d_right, d_arc))
	if scalar_input:
		distances = distances[0]

	if return_numpy:
		return distances
	return distances.tolist()

def visualize_ngon_side_distances(
	x=None,
	y=None,
	distances=None,
	n_sides=None,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	cmap="coolwarm",
	figsize=None,
	point_size=12,
	levels=64,
	show_polygon=True,
	show=True,
):
	"""
	Visualize signed distances for each side of a regular n-gon.

	Creates one subplot per side and renders that side's signed distance field using
	``contourf``, with the zero contour (distance = 0) overlaid.

	Args:
		x: scalar or array-like x coordinates.
		y: scalar or array-like y coordinates.
		distances: optional output from regular_ngon_side_signed_distances.
			If None, distances are computed internally.
		n_sides: number of sides (required when distances is None).
		radius: polygon circumradius.
		center: polygon center (cx, cy).
		rotation: angular offset in radians.
		cmap: matplotlib colormap name.
		figsize: optional figure size tuple.
		point_size: kept for backward compatibility (unused in contourf mode).
		levels: contourf levels (int or sequence).
		show_polygon: draw polygon outline on each subplot.
		show: if True, call plt.show().

	Returns:
		(fig, axes, distances_array)
	"""
	if x is None or y is None:
		x = np.linspace(-radius * 1.5, radius * 1.5, 200)
		y = np.linspace(-radius * 1.5, radius * 1.5, 200)
		x, y = np.meshgrid(x, y)
	x_arr = np.asarray(x, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)

	if x_arr.ndim == 1 and y_arr.ndim == 1:
		x_arr, y_arr = np.meshgrid(x_arr, y_arr)
	elif x_arr.shape != y_arr.shape:
		raise ValueError("x and y must have the same shape")

	if x_arr.ndim != 2 or y_arr.ndim != 2:
		raise ValueError("x and y must define a 2D grid for contourf plotting")

	grid_shape = x_arr.shape
	n_points = x_arr.size

	if distances is None:
		if n_sides is None:
			raise ValueError("n_sides must be provided when distances is None")
		dist_arr = regular_ngon_side_signed_distances(
			x_arr,
			y_arr,
			n_sides=n_sides,
			radius=radius,
			center=center,
			rotation=rotation,
			return_numpy=True,
		)
	else:
		if torch.is_tensor(distances):
			dist_arr = distances.detach().cpu().numpy().astype(np.float64, copy=False)
		else:
			dist_arr = np.asarray(distances, dtype=np.float64)

	if dist_arr.ndim == 3:
		if dist_arr.shape[:2] != grid_shape:
			raise ValueError("3D distances must have shape (ny, nx, n_sides)")
		dist_arr = dist_arr.reshape(n_points, dist_arr.shape[2])
	elif dist_arr.ndim == 2:
		if dist_arr.shape[0] != n_points:
			raise ValueError("distances first dimension must match number of grid points")
	else:
		raise ValueError("distances must be 2D (P, n_sides) or 3D (ny, nx, n_sides)")

	if n_sides is None:
		n_sides = int(dist_arr.shape[1])

	if dist_arr.shape[1] != n_sides:
		raise ValueError("distances second dimension must match n_sides")

	n_cols = int(np.ceil(np.sqrt(n_sides)))
	n_rows = int(np.ceil(n_sides / n_cols))

	if figsize is None:
		figsize = (4.2 * n_cols, 3.8 * n_rows)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
	axes_flat = axes.ravel()

	vmax = np.max(np.abs(dist_arr)) if dist_arr.size else 1.0
	vmin = -vmax

	if show_polygon:
		angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)
		cx, cy = center
		vx = cx + radius * np.cos(angles)
		vy = cy + radius * np.sin(angles)
		poly_x = np.append(vx, vx[0])
		poly_y = np.append(vy, vy[0])

	for i in range(n_sides):
		ax = axes_flat[i]
		z = dist_arr[:, i].reshape(grid_shape)
		cf = ax.contourf(
			x_arr,
			y_arr,
			z,
			levels=levels,
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
		)
		ax.contour(x_arr, y_arr, z, levels=[0.0], colors="r", linewidths=1.4)
		if show_polygon:
			ax.plot(poly_x, poly_y, "k-", linewidth=1.2)
		ax.set_title(f"Side {i}")
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

	for j in range(n_sides, n_rows * n_cols):
		axes_flat[j].axis("off")

	fig.tight_layout()

	if show:
		plt.show()

	return fig, axes, dist_arr


def evaluate_ngon_side_distances_on_grid(
	n_sides,
	resolution=128,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	extent=None,
	return_meshgrid=False,
):
	"""
	Evaluate the side-wise signed distance fields of a regular n-gon on an n x n grid.

	Returns:
		x_grid, y_grid, distances_grid
		where distances_grid has shape (ny, nx, n_sides).
	"""
	if n_sides < 3:
		raise ValueError("n_sides must be at least 3")
	if resolution < 2:
		raise ValueError("resolution must be at least 2")

	if extent is None:
		pad = radius * 1.2
		extent = (center[0] - pad, center[0] + pad, center[1] - pad, center[1] + pad)

	x_min, x_max, y_min, y_max = extent
	x_grid = np.linspace(x_min, x_max, resolution, dtype=np.float64)
	y_grid = np.linspace(y_min, y_max, resolution, dtype=np.float64)
	X, Y = np.meshgrid(x_grid, y_grid)
	distances = regular_ngon_side_signed_distances(
		X,
		Y,
		n_sides=n_sides,
		radius=radius,
		center=center,
		rotation=rotation,
		return_numpy=True,
	)
	if distances.ndim == 2:
		distances = distances.reshape(X.shape + (n_sides,))

	if return_meshgrid:
		return X, Y, distances
	return x_grid, y_grid, distances


def _bilinear_interpolate_and_gradient(x_points, y_points, x_grid, y_grid, field_grid):
	"""
	Bilinear interpolation and analytic gradient on a rectangular grid.

	field_grid must have shape (ny, nx).
	Returns interpolated values and gradients with shapes (N,), (N, 2).
	"""
	x_points = np.asarray(x_points, dtype=np.float64)
	y_points = np.asarray(y_points, dtype=np.float64)
	x_grid = np.asarray(x_grid, dtype=np.float64)
	y_grid = np.asarray(y_grid, dtype=np.float64)
	field_grid = np.asarray(field_grid, dtype=np.float64)

	if x_points.shape != y_points.shape:
		raise ValueError("x_points and y_points must have the same shape")
	if field_grid.shape != (y_grid.size, x_grid.size):
		raise ValueError("field_grid must have shape (ny, nx)")

	flat_x = x_points.ravel()
	flat_y = y_points.ravel()
	flat_x = np.clip(flat_x, x_grid[0], x_grid[-1])
	flat_y = np.clip(flat_y, y_grid[0], y_grid[-1])

	ix = np.searchsorted(x_grid, flat_x, side="right") - 1
	iy = np.searchsorted(y_grid, flat_y, side="right") - 1
	ix = np.clip(ix, 0, x_grid.size - 2)
	iy = np.clip(iy, 0, y_grid.size - 2)

	x0 = x_grid[ix]
	x1 = x_grid[ix + 1]
	y0 = y_grid[iy]
	y1 = y_grid[iy + 1]
	dx = x1 - x0
	dy = y1 - y0

	# Normalized coordinates inside the cell.
	tx = (flat_x - x0) / dx
	ty = (flat_y - y0) / dy

	f00 = field_grid[iy, ix]
	f10 = field_grid[iy, ix + 1]
	f01 = field_grid[iy + 1, ix]
	f11 = field_grid[iy + 1, ix + 1]

	values = (
		(1.0 - tx) * (1.0 - ty) * f00
		+ tx * (1.0 - ty) * f10
		+ (1.0 - tx) * ty * f01
		+ tx * ty * f11
	)

	dfdx = ((1.0 - ty) * (f10 - f00) + ty * (f11 - f01)) / dx
	dfdy = ((1.0 - tx) * (f01 - f00) + tx * (f11 - f10)) / dy
	gradients = np.column_stack((dfdx, dfdy))

	return values.reshape(x_points.shape), gradients.reshape(x_points.shape + (2,))


def evaluate_ngon_grid_interpolation_error(
	n_sides,
	resolution=128,
	num_points=5000,
	radius=1.0,
	center=(0.0, 0.0),
	rotation=0.0,
	extent=None,
	verbose=True,
	return_samples=False,
):
	"""
	Sample random points from inside the polygon and evaluate bilinearly interpolated
	side-wise distance fields from an n x n grid.

	For each side separately, computes:
	- mean absolute value error
	- maximum absolute value error
	- mean absolute gradient-norm error | |grad| - 1 |
	- maximum absolute gradient-norm error

	Returns a dictionary with global and per-side metrics.
	"""
	if num_points <= 0:
		raise ValueError("num_points must be positive")

	x_grid, y_grid, dist_grid = evaluate_ngon_side_distances_on_grid(
		n_sides=n_sides,
		resolution=resolution,
		radius=radius,
		center=center,
		rotation=rotation,
		extent=extent,
	)

	if extent is None:
		pad = radius * 1.2
		extent = (center[0] - pad, center[0] + pad, center[1] - pad, center[1] + pad)

	x_min, x_max, y_min, y_max = extent
	collected = []
	while sum(part.shape[0] for part in collected) < num_points:
		remaining = num_points - sum(part.shape[0] for part in collected)
		n_candidates = max(4096, remaining * 4)
		candidates = np.column_stack(
			(
				np.random.uniform(x_min, x_max, size=n_candidates),
				np.random.uniform(y_min, y_max, size=n_candidates),
			)
		)
		signed_dist = regular_ngon_side_signed_distances(
			candidates[:, 0],
			candidates[:, 1],
			n_sides=n_sides,
			radius=radius,
			center=center,
			rotation=rotation,
			use_sign=True,
			return_numpy=True,
		)
		inside_mask = np.all(signed_dist >= 0.0, axis=1)
		inside_points = candidates[inside_mask]
		if inside_points.size > 0:
			collected.append(inside_points)

	pts = np.vstack(collected)[:num_points]
	exact = regular_ngon_side_signed_distances(
		pts[:, 0],
		pts[:, 1],
		n_sides=n_sides,
		radius=radius,
		center=center,
		rotation=rotation,
		return_numpy=True,
	)

	if exact.ndim == 1:
		exact = exact[:, None]

	interp_values = np.empty_like(exact, dtype=np.float64)
	interp_grad_norm = np.empty_like(exact, dtype=np.float64)
	interp_grad_vec = np.empty((num_points, n_sides, 2), dtype=np.float64)

	for side_idx in range(n_sides):
		values_i, grads_i = _bilinear_interpolate_and_gradient(
			pts[:, 0],
			pts[:, 1],
			x_grid,
			y_grid,
			dist_grid[:, :, side_idx],
		)
		interp_values[:, side_idx] = values_i
		interp_grad_vec[:, side_idx, :] = grads_i
		interp_grad_norm[:, side_idx] = np.linalg.norm(grads_i, axis=1)

	value_abs_err = np.abs(interp_values - exact)
	grad_norm_abs_err = np.abs(interp_grad_norm - 1.0)

	results = {
		"num_points": int(num_points),
		"n_sides": int(n_sides),
		"grid_parameter_count": int(resolution * resolution * n_sides),
		"global": {
			"value_mean_abs_error": float(value_abs_err.mean()),
			"value_max_abs_error": float(value_abs_err.max()),
			"grad_norm_mean_abs_error": float(grad_norm_abs_err.mean()),
			"grad_norm_max_abs_error": float(grad_norm_abs_err.max()),
		},
		"per_side": [],
	}

	for side_idx in range(n_sides):
		results["per_side"].append(
			{
				"side": side_idx,
				"value_mean_abs_error": float(value_abs_err[:, side_idx].mean()),
				"value_max_abs_error": float(value_abs_err[:, side_idx].max()),
				"grad_norm_mean_abs_error": float(grad_norm_abs_err[:, side_idx].mean()),
				"grad_norm_max_abs_error": float(grad_norm_abs_err[:, side_idx].max()),
			}
		)

	if verbose:
		g = results["global"]
		print(
			f"Grid interpolation evaluation on {num_points} interior points: "
			f"value mean_abs={g['value_mean_abs_error']:.6e}, value max_abs={g['value_max_abs_error']:.6e}, "
			f"grad-norm mean_abs={g['grad_norm_mean_abs_error']:.6e}, grad-norm max_abs={g['grad_norm_max_abs_error']:.6e}"
		)
		print(
			f"Grid storage uses {results['grid_parameter_count']} scalar values "
			f"({resolution} x {resolution} x {n_sides})."
		)
		for side_res in results["per_side"]:
			print(
				f"  Side {side_res['side']}: "
				f"value(mean_abs={side_res['value_mean_abs_error']:.6e}, max_abs={side_res['value_max_abs_error']:.6e}) | "
				f"grad-norm(mean_abs={side_res['grad_norm_mean_abs_error']:.6e}, max_abs={side_res['grad_norm_max_abs_error']:.6e})"
			)

	if return_samples:
		results["samples"] = {
			"points": pts,
			"exact": exact,
			"interp_values": interp_values,
			"interp_gradients": interp_grad_vec,
			"interp_grad_norm": interp_grad_norm,
		}

	return results

if __name__ == "__main__":
    #visualize_ngon_side_distances(n_sides=5, radius=1.0, rotation=np.pi / 5)
	evaluate_ngon_grid_interpolation_error(n_sides=5, resolution=200, num_points=10000, radius=1.0, rotation=np.pi / 5)
	