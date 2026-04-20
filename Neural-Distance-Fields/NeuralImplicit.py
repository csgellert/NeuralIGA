import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import network_defs as net_defs
import multi_SDF as SDF
from matplotlib.animation import FuncAnimation

def get_gradient_error(model, grds_gt, pts,metric='L1'):
    pts.requires_grad_(True)
    pred = model(pts)
    grads = torch.autograd.grad(outputs=pred, inputs=pts,
                                grad_outputs=torch.ones_like(pred),
                                create_graph=True, retain_graph=True)[0]
    # cosine similarity of grads between gt ang model predictions
    cos = nn.CosineSimilarity(dim=1, eps=1e-16)
    similarity = cos(grads, grds_gt)
    similarity_error = 1-similarity
    lengths = torch.norm(grads, dim=1)
    if metric == 'L1':
        mean_similarity = torch.mean(torch.abs(similarity_error)).item()
        length_error = torch.mean(torch.abs(lengths - 1)).item()
    elif metric == 'L_inf':
        mean_similarity = torch.max(similarity_error).item()
        length_error = torch.max(torch.abs(lengths - 1)).item()
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    return length_error, mean_similarity

def generate_data(batch_size, fun_num=1, device=None, data_gen_params={}):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if fun_num == 1: #Ngon
        n_sides = data_gen_params.get('n_sides', 5)
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        rotation = data_gen_params.get('rotation', 0.0)

        # Rejection sampling in the bounding square, keeping only points inside the polygon.
        center_t = torch.tensor(center, device=device)
        collected = []
        num_collected = 0
        while num_collected < batch_size:
            n_missing = batch_size - num_collected
            n_candidates = max(1024, n_missing * 2)
            candidates = torch.rand(n_candidates, 2, device=device) * radius * 2 - radius + center_t

            signed_dist = SDF.regular_ngon_side_signed_distances(
                candidates[:, 0],
                candidates[:, 1],
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=False,
            )
            if torch.is_tensor(signed_dist):
                inside_mask = torch.all(signed_dist >= 0.0, dim=1)
            else:
                signed_dist_np = np.asarray(signed_dist)
                inside_mask_np = np.all(signed_dist_np >= 0.0, axis=1)
                inside_mask = torch.as_tensor(inside_mask_np, device=device, dtype=torch.bool)
            inside_points = candidates[inside_mask]

            if inside_points.shape[0] > 0:
                take_n = min(n_missing, inside_points.shape[0])
                collected.append(inside_points[:take_n])
                num_collected += take_n

        pts = torch.cat(collected, dim=0)
        target = SDF.regular_ngon_side_signed_distances(
            pts[:, 0], pts[:, 1], n_sides=n_sides, radius=radius, center=center, rotation=rotation, return_numpy=False
        )
        #apply mobius transformation for target y = (1+x)/(1-x)

        #target = (np.ones_like(target) - target) / (np.ones_like(target) + target)

        return pts, target
    elif fun_num == 2:  # Semicircle + isosceles triangle union
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        apex = data_gen_params.get('apex', (center[0], center[1] - radius))

        cx, cy = center
        ax, ay = apex
        # Sampling bbox of the union domain.
        x_min = min(cx - radius, cx + radius, ax) - 0.05 * radius
        x_max = max(cx - radius, cx + radius, ax) + 0.05 * radius
        y_min = min(cy, ay, cy + radius) - 0.05 * radius
        y_max = max(cy, ay, cy + radius) + 0.05 * radius

        collected = []
        num_collected = 0
        while num_collected < batch_size:
            n_missing = batch_size - num_collected
            n_candidates = max(2048, n_missing * 3)
            candidates = torch.empty(n_candidates, 2, device=device)
            candidates[:, 0] = x_min + (x_max - x_min) * torch.rand(n_candidates, device=device)
            candidates[:, 1] = y_min + (y_max - y_min) * torch.rand(n_candidates, device=device)

            inside_mask = SDF.is_inside_semicircle_triangle_union(
                candidates[:, 0],
                candidates[:, 1],
                radius=radius,
                center=center,
                apex=apex,
            )
            if not torch.is_tensor(inside_mask):
                inside_mask = torch.as_tensor(inside_mask, device=device, dtype=torch.bool)

            inside_points = candidates[inside_mask]
            if inside_points.shape[0] > 0:
                take_n = min(n_missing, inside_points.shape[0])
                collected.append(inside_points[:take_n])
                num_collected += take_n

        pts = torch.cat(collected, dim=0)
        target = SDF.semicircle_triangle_side_distances(
            pts[:, 0],
            pts[:, 1],
            radius=radius,
            center=center,
            apex=apex,
            return_numpy=False,
        )
        return pts, target
    else:
        raise NotImplementedError(f"Data generation for fun_num={fun_num} not implemented in this snippet.")


def generate_domain_boundary_points(num_points, fun_num=1, device=None, data_gen_params={}, return_side_indices=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if fun_num == 1:
        n_sides = data_gen_params.get('n_sides', 5)
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        rotation = data_gen_params.get('rotation', 0.0)

        if n_sides < 3:
            raise ValueError("n_sides must be at least 3")

        cx, cy = center
        angles = rotation + (2.0 * np.pi / n_sides) * np.arange(n_sides, dtype=np.float64)
        vx = cx + radius * np.cos(angles)
        vy = cy + radius * np.sin(angles)
        vertices = torch.as_tensor(np.column_stack((vx, vy)), device=device, dtype=torch.float32)

        edge_idx = torch.randint(0, n_sides, (num_points,), device=device)
        t = torch.rand(num_points, device=device)

        start = vertices[edge_idx]
        end = vertices[(edge_idx + 1) % n_sides]
        pts = start + t.unsqueeze(1) * (end - start)
        if return_side_indices:
            return pts, edge_idx
        return pts
    if fun_num == 2:
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        apex = data_gen_params.get('apex', (center[0], center[1] - radius))

        cx, cy = center
        ax, ay = apex
        a = torch.tensor([cx - radius, cy], device=device, dtype=torch.float32)
        b = torch.tensor([cx + radius, cy], device=device, dtype=torch.float32)
        c = torch.tensor([ax, ay], device=device, dtype=torch.float32)

        side_idx = torch.randint(0, 3, (num_points,), device=device)
        t = torch.rand(num_points, device=device)
        pts = torch.empty(num_points, 2, device=device, dtype=torch.float32)

        # Side 0: left line A -> apex
        m0 = side_idx == 0
        if torch.any(m0):
            tt = t[m0].unsqueeze(1)
            pts[m0] = a.unsqueeze(0) + tt * (c - a).unsqueeze(0)

        # Side 1: right line apex -> B
        m1 = side_idx == 1
        if torch.any(m1):
            tt = t[m1].unsqueeze(1)
            pts[m1] = c.unsqueeze(0) + tt * (b - c).unsqueeze(0)

        # Side 2: upper semicircle arc A -> B
        m2 = side_idx == 2
        if torch.any(m2):
            theta = np.pi * (1.0 - t[m2])
            pts[m2, 0] = cx + radius * torch.cos(theta)
            pts[m2, 1] = cy + radius * torch.sin(theta)

        if return_side_indices:
            return pts, side_idx
        return pts

    raise NotImplementedError(f"Boundary point generation for fun_num={fun_num} is not implemented")


def train_model_simple(model, num_epochs=100, batch_size=10000, fun_num=1, *, device=None,
                       crt=nn.L1Loss(), use_scheduler=False, data_gen_params={}, 
                       boundary_error_coeff=0.0,
                       eikon_coeff=0.0,
                       hotspot_coeff=0.0,
                       pred_coeff=1.0,
                       hotspot_params={}):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion = crt
    report_interval = max(1, num_epochs // 10)

    if not hasattr(model, 'loss_history'):
        model.loss_history = []

    for epoch in range(num_epochs):
        pts, target = generate_data(
            batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params
        )

        pred = model(pts)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, device=device, dtype=pred.dtype)
        else:
            target = target.to(device=device, dtype=pred.dtype)
        if target.ndim == 1 and pred.ndim == 2 and pred.shape[1] == 1:
            target = target.unsqueeze(1)
        pred_loss = criterion(pred, target)
        loss = pred_coeff*pred_loss
        if boundary_error_coeff > 0.0:
            bnd_data = generate_domain_boundary_points(
                batch_size, fun_num=fun_num, device=device, data_gen_params=data_gen_params, return_side_indices=True
            )
            if isinstance(bnd_data, tuple):
                bnd_pts, bnd_side_idx = bnd_data
            else:
                bnd_pts, bnd_side_idx = bnd_data, None
            bnd_pred = model(bnd_pts)
            if bnd_pred.ndim == 1:
                bnd_pred = bnd_pred.unsqueeze(1)

            n_boundary_sides = int(data_gen_params.get('n_sides', bnd_pred.shape[1]))
            if bnd_pred.shape[1] == 1 or bnd_side_idx is None or bnd_pred.shape[1] != n_boundary_sides:
                bnd_target = torch.zeros_like(bnd_pred)
                boundary_loss = criterion(bnd_pred, bnd_target)
            else:
                # Only the output channel corresponding to the touched side should be zero.
                selected_pred = bnd_pred[torch.arange(bnd_pred.shape[0], device=device), bnd_side_idx]
                boundary_loss = criterion(selected_pred, torch.zeros_like(selected_pred))
            loss += boundary_error_coeff * boundary_loss

        if eikon_coeff > 0.0 or hotspot_coeff > 0.0:
            pts_reg = pts.detach().clone().requires_grad_(True)
            pred_reg = model(pts_reg)
            if pred_reg.ndim == 1:
                pred_reg = pred_reg.unsqueeze(1)

            lambda_hotspot = hotspot_params.get('lambda', 0.1)
            eikonal_terms = []
            hotspot_terms = []

            for i in range(pred_reg.shape[1]):
                grads_i = torch.autograd.grad(
                    outputs=pred_reg[:, i].sum(),
                    inputs=pts_reg,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grad_norm_i = torch.norm(grads_i, dim=1)

                if eikon_coeff > 0.0:
                    eikonal_terms.append(criterion(grad_norm_i, torch.ones_like(grad_norm_i)))

                if hotspot_coeff > 0.0:
                    phi_i = pred_reg[:, i]
                    hotspot_i = torch.mean(0.5 * torch.exp(-2 * lambda_hotspot * torch.abs(phi_i)) * (grad_norm_i ** 2 + 1.0))
                    hotspot_terms.append(hotspot_i)

            if eikon_coeff > 0.0 and len(eikonal_terms) > 0:
                eikonal_loss = torch.stack(eikonal_terms).mean()
                loss += eikon_coeff * eikonal_loss

            if hotspot_coeff > 0.0 and len(hotspot_terms) > 0:
                hotspot_loss = torch.stack(hotspot_terms).mean()
                loss += hotspot_coeff * hotspot_loss

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if use_scheduler and model.lr_scheduler is not None:
            model.lr_scheduler.step()

        model.loss_history.append(loss.item())

        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch}], Loss: {loss.item()}, Acc: {pred_loss.item()}")




def _evaluate_ground_truth_on_points(pts, fun_num=1, data_gen_params={}):
    if fun_num == 1:
        n_sides = data_gen_params.get('n_sides', 5)
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        rotation = data_gen_params.get('rotation', 0.0)
        gt_np = SDF.regular_ngon_side_signed_distances(
            pts[:, 0].detach().cpu().numpy(),
            pts[:, 1].detach().cpu().numpy(),
            n_sides=n_sides,
            radius=radius,
            center=center,
            rotation=rotation,
            return_numpy=True,
        )
        gt = torch.as_tensor(gt_np, device=pts.device, dtype=pts.dtype)
        if gt.ndim == 1:
            gt = gt.unsqueeze(1)
        return gt
    if fun_num == 2:
        radius = data_gen_params.get('radius', 0.5)
        center = data_gen_params.get('center', (0.0, 0.0))
        apex = data_gen_params.get('apex', (center[0], center[1] - radius))
        gt_np = SDF.semicircle_triangle_side_distances(
            pts[:, 0].detach().cpu().numpy(),
            pts[:, 1].detach().cpu().numpy(),
            radius=radius,
            center=center,
            apex=apex,
            return_numpy=True,
        )
        gt = torch.as_tensor(gt_np, device=pts.device, dtype=pts.dtype)
        if gt.ndim == 1:
            gt = gt.unsqueeze(1)
        return gt
    raise NotImplementedError(f"Ground-truth evaluation for fun_num={fun_num} is not implemented")


def evaluate_model_random_points(
    model,
    function_case,
    N,
    *,
    device=None,
    data_gen_params={},
    per_side_report=False,
    verbose=True,
):
    """
    Evaluate model quality on N random points for the selected function case.

    Metrics are computed for value prediction, gradient-norm consistency
    (Eikonal-type error, where |grad(phi)| should be 1), and Hessian-diagonal
    consistency (where d2phi/dx2 and d2phi/dy2 should both be 0).

    Args:
        model: torch model that maps (N, 2) -> (N, n_fields).
        function_case: integer identifier of the target function (same meaning as fun_num).
        N: number of random evaluation points.
        device: torch device.
        data_gen_params: parameters forwarded to point/target generation.
        per_side_report: if True, report metrics for each output side separately.
        verbose: if True, print formatted metrics.

    Returns:
        Dictionary with global metrics and optionally side-wise metrics.

    Example:
        results = evaluate_model_random_points(
            model,
            function_case=1,
            N=20000,
            data_gen_params={"n_sides": 5, "radius": 0.5},
            per_side_report=True,
        )
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    pts, target = generate_data(
        int(N),
        fun_num=function_case,
        device=device,
        data_gen_params=data_gen_params,
    )

    if not torch.is_tensor(target):
        target = torch.as_tensor(target, device=device, dtype=pts.dtype)
    else:
        target = target.to(device=device, dtype=pts.dtype)

    pts_eval = pts.detach().clone().requires_grad_(True)
    pred = model(pts_eval)

    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}")

    err = pred - target
    abs_err = torch.abs(err)

    # Value errors across all points and all sides.
    mean_abs_error = abs_err.mean().item()
    max_abs_error = abs_err.max().item()
    l2_error = torch.sqrt(torch.mean(err ** 2)).item()

    n_fields = pred.shape[1]
    grad_norm_error_per_side = []
    hessian_diag_error_per_side = []
    mean_abs_error_per_side = []
    max_abs_error_per_side = []
    l2_error_per_side = []

    for i in range(n_fields):
        grad_i = torch.autograd.grad(
            outputs=pred[:, i].sum(),
            inputs=pts_eval,
            create_graph=True,
            # We need this graph for second-derivative calls below.
            retain_graph=True,
        )[0]
        grad_norm_i = torch.norm(grad_i, dim=1)
        grad_norm_err_i = torch.abs(grad_norm_i - 1.0)
        grad_norm_error_per_side.append(grad_norm_err_i)

        d2phi_dx2_i = torch.autograd.grad(
            outputs=grad_i[:, 0].sum(),
            inputs=pts_eval,
            create_graph=False,
            retain_graph=True,
        )[0][:, 0]
        d2phi_dy2_i = torch.autograd.grad(
            outputs=grad_i[:, 1].sum(),
            inputs=pts_eval,
            create_graph=False,
            # Keep graph for next side's first-derivative backward pass.
            retain_graph=(i < n_fields - 1),
        )[0][:, 1]

        # Ground truth Hessian diagonal is zero for signed distance to linear/arc sides.
        hessian_diag_err_i = torch.stack([
            torch.abs(d2phi_dx2_i),
            torch.abs(d2phi_dy2_i),
        ], dim=1)
        hessian_diag_error_per_side.append(hessian_diag_err_i)

        err_i = err[:, i]
        mean_abs_error_per_side.append(torch.mean(torch.abs(err_i)).item())
        max_abs_error_per_side.append(torch.max(torch.abs(err_i)).item())
        l2_error_per_side.append(torch.sqrt(torch.mean(err_i ** 2)).item())

    grad_norm_error = torch.stack(grad_norm_error_per_side, dim=1)
    grad_mean_abs_error = grad_norm_error.mean().item()
    grad_max_abs_error = grad_norm_error.max().item()
    grad_l2_error = torch.sqrt(torch.mean(grad_norm_error ** 2)).item()

    hessian_diag_error = torch.stack(hessian_diag_error_per_side, dim=1)
    hessian_diag_mean_abs_error = hessian_diag_error.mean().item()
    hessian_diag_max_abs_error = hessian_diag_error.max().item()
    hessian_diag_l2_error = torch.sqrt(torch.mean(hessian_diag_error ** 2)).item()

    results = {
        "N": int(N),
        "function_case": int(function_case),
        "global": {
            "value_mean_abs_error": mean_abs_error,
            "value_max_abs_error": max_abs_error,
            "value_l2_error": l2_error,
            "grad_norm_mean_abs_error": grad_mean_abs_error,
            "grad_norm_max_abs_error": grad_max_abs_error,
            "grad_norm_l2_error": grad_l2_error,
            "hessian_diag_mean_abs_error": hessian_diag_mean_abs_error,
            "hessian_diag_max_abs_error": hessian_diag_max_abs_error,
            "hessian_diag_l2_error": hessian_diag_l2_error,
        },
    }

    if per_side_report:
        side_metrics = []
        for i in range(n_fields):
            gerr_i = grad_norm_error[:, i]
            herr_i = hessian_diag_error[:, i, :]
            side_metrics.append(
                {
                    "side": i,
                    "value_mean_abs_error": mean_abs_error_per_side[i],
                    "value_max_abs_error": max_abs_error_per_side[i],
                    "value_l2_error": l2_error_per_side[i],
                    "grad_norm_mean_abs_error": gerr_i.mean().item(),
                    "grad_norm_max_abs_error": gerr_i.max().item(),
                    "grad_norm_l2_error": torch.sqrt(torch.mean(gerr_i ** 2)).item(),
                    "hessian_diag_mean_abs_error": herr_i.mean().item(),
                    "hessian_diag_max_abs_error": herr_i.max().item(),
                    "hessian_diag_l2_error": torch.sqrt(torch.mean(herr_i ** 2)).item(),
                    "hessian_xx_mean_abs_error": herr_i[:, 0].mean().item(),
                    "hessian_yy_mean_abs_error": herr_i[:, 1].mean().item(),
                }
            )
        results["per_side"] = side_metrics

    if verbose:
        g = results["global"]
        print(f"Model evaluation on N={results['N']} random points (function_case={results['function_case']}):")
        print(
            f"  Value errors: mean_abs={g['value_mean_abs_error']:.6e}, "
            f"max_abs={g['value_max_abs_error']:.6e}, L2={g['value_l2_error']:.6e}"
        )
        print(
            f"  Grad-norm errors (|grad|-1): mean_abs={g['grad_norm_mean_abs_error']:.6e}, "
            f"max_abs={g['grad_norm_max_abs_error']:.6e}, L2={g['grad_norm_l2_error']:.6e}"
        )
        print(
            f"  Hessian-diagonal errors (target 0): mean_abs={g['hessian_diag_mean_abs_error']:.6e}, "
            f"max_abs={g['hessian_diag_max_abs_error']:.6e}, L2={g['hessian_diag_l2_error']:.6e}"
        )
        if per_side_report and "per_side" in results:
            for sm in results["per_side"]:
                print(
                    f"  Side {sm['side']}: "
                    f"value(mean_abs={sm['value_mean_abs_error']:.6e}, "
                    f"max_abs={sm['value_max_abs_error']:.6e}, L2={sm['value_l2_error']:.6e}) | "
                    f"grad(|grad|-1 mean_abs={sm['grad_norm_mean_abs_error']:.6e}, "
                    f"max_abs={sm['grad_norm_max_abs_error']:.6e}, L2={sm['grad_norm_l2_error']:.6e}) | "
                    f"hdiag(mean_abs={sm['hessian_diag_mean_abs_error']:.6e}, "
                    f"max_abs={sm['hessian_diag_max_abs_error']:.6e}, L2={sm['hessian_diag_l2_error']:.6e}, "
                    f"xx_mean={sm['hessian_xx_mean_abs_error']:.6e}, "
                    f"yy_mean={sm['hessian_yy_mean_abs_error']:.6e})"
                )

    return results


def plot_nn_distance_fields(model, resolution=200, extent=(-1.0, 1.0, -1.0, 1.0), *, device=None,
                            cmap='coolwarm', levels=64, show_zero_level=True, show=True,
                            mask_outside_domain=True, fun_num=1, data_gen_params={},
                            supersample_factor=2):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    if supersample_factor < 1:
        raise ValueError("supersample_factor must be >= 1")

    x_min, x_max, y_min, y_max = extent
    render_resolution = int(resolution * supersample_factor)
    X, Y = torch.meshgrid(
        torch.linspace(x_min, x_max, render_resolution, device=device),
        torch.linspace(y_min, y_max, render_resolution, device=device),
    )
    grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)

    with torch.no_grad():
        pred = model(grid_points)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    pred_np = pred.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    n_fields = pred_np.shape[1]

    if mask_outside_domain:
        if fun_num == 1:
            n_sides = data_gen_params.get('n_sides', 5)
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            rotation = data_gen_params.get('rotation', 0.0)

            signed_dist = SDF.regular_ngon_side_signed_distances(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=True,
            )
            signed_dist_np = np.asarray(signed_dist)
            if signed_dist_np.ndim == 1:
                signed_dist_np = signed_dist_np[:, None]
            domain_mask = np.all(signed_dist_np >= 0.0, axis=1).reshape(render_resolution, render_resolution)
        elif fun_num == 2:
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            apex = data_gen_params.get('apex', (center[0], center[1] - radius))
            inside = SDF.is_inside_semicircle_triangle_union(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                radius=radius,
                center=center,
                apex=apex,
                return_numpy=True,
            )
            domain_mask = np.asarray(inside, dtype=bool).reshape(render_resolution, render_resolution)
        else:
            raise NotImplementedError(f"Domain masking for fun_num={fun_num} is not implemented")
    else:
        domain_mask = np.ones((render_resolution, render_resolution), dtype=bool)

    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(n_fields):
        ax = axes_flat[i]
        z = pred_np[:, i].reshape(render_resolution, render_resolution)
        z_masked = np.ma.array(z, mask=~domain_mask)
        # corner_mask softens mask transitions near the boundary triangles.
        if show_zero_level:
            ax.contour(X_np, Y_np, z_masked, levels=[0.0], colors='k', linewidths=1.2)
        cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap=cmap, corner_mask=True)
        ax.set_title(f'Predicted distance field {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_fields, n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, pred_np


def plot_ground_truth_distance_fields(resolution=200, extent=(-1.0, 1.0, -1.0, 1.0), *, device=None,
                                      cmap='coolwarm', levels=64, show_zero_level=True, show=True,
                                      mask_outside_domain=True, fun_num=1, data_gen_params={}):
    """
    Plot analytical ground-truth distance fields for the selected test case.

    This mirrors ``plot_nn_distance_fields`` but evaluates target fields directly,
    without a neural model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_min, x_max, y_min, y_max = extent
    X, Y = torch.meshgrid(
        torch.linspace(x_min, x_max, resolution, device=device),
        torch.linspace(y_min, y_max, resolution, device=device),
    )
    grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)

    target = _evaluate_ground_truth_on_points(grid_points, fun_num=fun_num, data_gen_params=data_gen_params)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    target_np = target.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    n_fields = target_np.shape[1]

    if mask_outside_domain:
        if fun_num == 1:
            n_sides = data_gen_params.get('n_sides', 5)
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            rotation = data_gen_params.get('rotation', 0.0)

            signed_dist = SDF.regular_ngon_side_signed_distances(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=True,
            )
            signed_dist_np = np.asarray(signed_dist)
            if signed_dist_np.ndim == 1:
                signed_dist_np = signed_dist_np[:, None]
            domain_mask = np.all(signed_dist_np >= 0.0, axis=1).reshape(resolution, resolution)
        elif fun_num == 2:
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            apex = data_gen_params.get('apex', (center[0], center[1] - radius))
            inside = SDF.is_inside_semicircle_triangle_union(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                radius=radius,
                center=center,
                apex=apex,
                return_numpy=True,
            )
            domain_mask = np.asarray(inside, dtype=bool).reshape(resolution, resolution)
        else:
            raise NotImplementedError(f"Domain masking for fun_num={fun_num} is not implemented")
    else:
        domain_mask = np.ones((resolution, resolution), dtype=bool)

    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(n_fields):
        ax = axes_flat[i]
        z = target_np[:, i].reshape(resolution, resolution)
        z_masked = np.ma.array(z, mask=~domain_mask)
        if show_zero_level:
            ax.contour(X_np, Y_np, z_masked, levels=[0.0], colors='k', linewidths=1.2)
        cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap=cmap, corner_mask=True)
        ax.set_title(f'Ground-truth distance field {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_fields, n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, target_np


def plot_nn_prediction_error(model, fun_num=1, resolution=200, extent=(-1.0, 1.0, -1.0, 1.0), *, device=None,
                             data_gen_params={}, metric='abs', levels=64, show_zero_level=True, show=True,
                             mask_outside_domain=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    x_min, x_max, y_min, y_max = extent
    X, Y = torch.meshgrid(
        torch.linspace(x_min, x_max, resolution, device=device),
        torch.linspace(y_min, y_max, resolution, device=device),
    )
    grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)

    with torch.no_grad():
        pred = model(grid_points)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    target = _evaluate_ground_truth_on_points(grid_points, fun_num=fun_num, data_gen_params=data_gen_params)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}")

    if metric == 'abs':
        err = torch.abs(pred - target)
        cmap = 'hot'
    elif metric == 'squared':
        err = (pred - target) ** 2
        cmap = 'hot'
    elif metric == 'signed':
        err = pred - target
        cmap = 'coolwarm'
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")

    err_np = err.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    n_fields = err_np.shape[1]

    if mask_outside_domain:
        if fun_num == 1:
            n_sides = data_gen_params.get('n_sides', 5)
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            rotation = data_gen_params.get('rotation', 0.0)

            signed_dist = SDF.regular_ngon_side_signed_distances(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=True,
            )
            signed_dist_np = np.asarray(signed_dist)
            if signed_dist_np.ndim == 1:
                signed_dist_np = signed_dist_np[:, None]
            domain_mask = np.all(signed_dist_np >= 0.0, axis=1).reshape(resolution, resolution)
        elif fun_num == 2:
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            apex = data_gen_params.get('apex', (center[0], center[1] - radius))
            inside = SDF.is_inside_semicircle_triangle_union(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                radius=radius,
                center=center,
                apex=apex,
                return_numpy=True,
            )
            domain_mask = np.asarray(inside, dtype=bool).reshape(resolution, resolution)
        else:
            raise NotImplementedError(f"Domain masking for fun_num={fun_num} is not implemented")
    else:
        domain_mask = np.ones((resolution, resolution), dtype=bool)

    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(n_fields):
        ax = axes_flat[i]
        z = err_np[:, i].reshape(resolution, resolution)
        z_masked = np.ma.array(z, mask=~domain_mask)
        cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap=cmap, corner_mask=True)
        if show_zero_level:
            ax.contour(X_np, Y_np, z_masked, levels=[0.0], colors='k', linewidths=1.2)
        ax.set_title(f'Prediction error field {i} ({metric})')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_fields, n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, err_np


def plot_local_per_side_gradient_error(model, resolution=200, extent=(-1.0, 1.0, -1.0, 1.0), *, device=None,
                                       fun_num=1, data_gen_params={}, levels=64, cmap='hot',
                                       mask_outside_domain=True, show=True):
    """
    Plot local distribution of per-side gradient-norm error maps.

    For each output field phi_i, this function computes:
        error_i(x, y) = | ||grad(phi_i)|| - 1 |
    and displays one contour plot per side.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    x_min, x_max, y_min, y_max = extent
    X, Y = torch.meshgrid(
        torch.linspace(x_min, x_max, resolution, device=device),
        torch.linspace(y_min, y_max, resolution, device=device),
    )
    grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1).requires_grad_(True)

    pred = model(grid_points)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    n_fields = pred.shape[1]
    grad_norm_error_maps = []

    for i in range(n_fields):
        grads_i = torch.autograd.grad(
            outputs=pred[:, i].sum(),
            inputs=grid_points,
            create_graph=False,
            retain_graph=(i < n_fields - 1),
        )[0]
        grad_norm_i = torch.norm(grads_i, dim=1)
        grad_err_i = torch.abs(grad_norm_i - 1.0)
        grad_norm_error_maps.append(grad_err_i.reshape(resolution, resolution).detach().cpu().numpy())

    grad_norm_error_np = np.stack(grad_norm_error_maps, axis=2)
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    if mask_outside_domain:
        if fun_num == 1:
            n_sides = data_gen_params.get('n_sides', 5)
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            rotation = data_gen_params.get('rotation', 0.0)
            signed_dist = SDF.regular_ngon_side_signed_distances(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=True,
            )
            signed_dist_np = np.asarray(signed_dist)
            if signed_dist_np.ndim == 1:
                signed_dist_np = signed_dist_np[:, None]
            domain_mask = np.all(signed_dist_np >= 0.0, axis=1).reshape(resolution, resolution)
        elif fun_num == 2:
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            apex = data_gen_params.get('apex', (center[0], center[1] - radius))
            inside = SDF.is_inside_semicircle_triangle_union(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                radius=radius,
                center=center,
                apex=apex,
                return_numpy=True,
            )
            domain_mask = np.asarray(inside, dtype=bool).reshape(resolution, resolution)
        else:
            raise NotImplementedError(f"Domain masking for fun_num={fun_num} is not implemented")
    else:
        domain_mask = np.ones((resolution, resolution), dtype=bool)

    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    vmax = float(np.max(grad_norm_error_np)) if grad_norm_error_np.size else 1.0

    for i in range(n_fields):
        ax = axes_flat[i]
        z = grad_norm_error_np[:, :, i]
        z_masked = np.ma.array(z, mask=~domain_mask)
        cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f'Grad-norm error side {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_fields, n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes, grad_norm_error_np


def plot_local_per_side_second_derivative(model, resolution=200, extent=(-1.0, 1.0, -1.0, 1.0), *, device=None,
                                          fun_num=1, data_gen_params={}, levels=64, cmap='coolwarm',
                                          component='laplacian', absolute=False,
                                          mask_outside_domain=True, show=True):
    """
    Plot local distribution of per-side second derivatives.

    For each output field phi_i, this function computes one Hessian-diagonal
    component map and displays one contour plot per side:
        component='xx' -> d2phi_i/dx2
        component='yy' -> d2phi_i/dy2
        component='laplacian' -> d2phi_i/dx2 + d2phi_i/dy2

    Since analytical target is zero, ``absolute=True`` is useful to view local
    magnitude of deviation from zero.
    """
    if component not in ('xx', 'yy', 'laplacian'):
        raise ValueError("component must be one of: 'xx', 'yy', 'laplacian'")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    x_min, x_max, y_min, y_max = extent
    X, Y = torch.meshgrid(
        torch.linspace(x_min, x_max, resolution, device=device),
        torch.linspace(y_min, y_max, resolution, device=device),
    )
    grid_points = torch.stack([X.ravel(), Y.ravel()], dim=-1).requires_grad_(True)

    pred = model(grid_points)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    n_fields = pred.shape[1]
    second_derivative_maps = []

    for i in range(n_fields):
        grads_i = torch.autograd.grad(
            outputs=pred[:, i].sum(),
            inputs=grid_points,
            create_graph=True,
            retain_graph=True,
        )[0]

        if component == 'xx':
            second_i = torch.autograd.grad(
                outputs=grads_i[:, 0].sum(),
                inputs=grid_points,
                create_graph=False,
                retain_graph=(i < n_fields - 1),
            )[0][:, 0]
        elif component == 'yy':
            second_i = torch.autograd.grad(
                outputs=grads_i[:, 1].sum(),
                inputs=grid_points,
                create_graph=False,
                retain_graph=(i < n_fields - 1),
            )[0][:, 1]
        else:  # laplacian
            d2phi_dx2_i = torch.autograd.grad(
                outputs=grads_i[:, 0].sum(),
                inputs=grid_points,
                create_graph=False,
                retain_graph=True,
            )[0][:, 0]
            d2phi_dy2_i = torch.autograd.grad(
                outputs=grads_i[:, 1].sum(),
                inputs=grid_points,
                create_graph=False,
                retain_graph=(i < n_fields - 1),
            )[0][:, 1]
            second_i = d2phi_dx2_i + d2phi_dy2_i

        if absolute:
            second_i = torch.abs(second_i)

        second_derivative_maps.append(second_i.reshape(resolution, resolution).detach().cpu().numpy())

    second_derivative_np = np.stack(second_derivative_maps, axis=2)
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    if mask_outside_domain:
        if fun_num == 1:
            n_sides = data_gen_params.get('n_sides', 5)
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            rotation = data_gen_params.get('rotation', 0.0)
            signed_dist = SDF.regular_ngon_side_signed_distances(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                n_sides=n_sides,
                radius=radius,
                center=center,
                rotation=rotation,
                use_sign=True,
                return_numpy=True,
            )
            signed_dist_np = np.asarray(signed_dist)
            if signed_dist_np.ndim == 1:
                signed_dist_np = signed_dist_np[:, None]
            domain_mask = np.all(signed_dist_np >= 0.0, axis=1).reshape(resolution, resolution)
        elif fun_num == 2:
            radius = data_gen_params.get('radius', 0.5)
            center = data_gen_params.get('center', (0.0, 0.0))
            apex = data_gen_params.get('apex', (center[0], center[1] - radius))
            inside = SDF.is_inside_semicircle_triangle_union(
                grid_points[:, 0].detach().cpu().numpy(),
                grid_points[:, 1].detach().cpu().numpy(),
                radius=radius,
                center=center,
                apex=apex,
                return_numpy=True,
            )
            domain_mask = np.asarray(inside, dtype=bool).reshape(resolution, resolution)
        else:
            raise NotImplementedError(f"Domain masking for fun_num={fun_num} is not implemented")
    else:
        domain_mask = np.ones((resolution, resolution), dtype=bool)

    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    vmax = float(np.max(np.abs(second_derivative_np))) if second_derivative_np.size else 1.0

    for i in range(n_fields):
        ax = axes_flat[i]
        z = second_derivative_np[:, :, i]
        z_masked = np.ma.array(z, mask=~domain_mask)
        if absolute:
            cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap='hot', vmin=0.0, vmax=vmax)
        else:
            cf = ax.contourf(X_np, Y_np, z_masked, levels=levels, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f'Second derivative side {i} ({component})')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_fields, n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes, second_derivative_np


def plot_model_weight_per_layer_hyst(model):
    # Plot weight histograms for each layer in the model
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.cpu().numpy().flatten()
            plt.figure(figsize=(8, 4))
            plt.hist(weights, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution for Layer {i} ({layer.__class__.__name__})')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
        elif isinstance(layer, net_defs.SineLayer):
            weights = layer.linear.weight.data.cpu().numpy().flatten()
            plt.figure(figsize=(8, 4))
            plt.hist(weights, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution for SineLayer {i}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()



if __name__ == "__main__":
    pts, trgt = generate_data(batch_size=2, fun_num=1, device=torch.device('cpu'), data_gen_params={'n_sides': 5, 'radius': 0.5, 'center': (0.0, 0.0), 'rotation': 0.0})
    print("Sample points:\n", pts)
    print("Target distances:\n", trgt)