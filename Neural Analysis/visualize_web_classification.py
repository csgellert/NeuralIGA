"""
Visualization utilities for WEB-spline element and basis-function classification.

Provides functions to visualize:
  - Domain (SDF level set)
  - Element classification (inner/boundary/outer)
  - B-spline classification (inner/outer)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import FEM_WEB
import mesh


def visualize_domain_and_classification(
    model,
    p, q,
    knotvector_x, knotvector_y,
    xDivision, yDivision,
    element_types=None,
    bspline_classification=None,
    figsize=(16, 12),
):
    """
    Visualize the domain, element classification, and B-spline classification.

    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model representing the geometry (SDF)
    p, q : int
        B-spline degrees
    knotvector_x, knotvector_y : array-like
        Knot vectors
    xDivision, yDivision : int
        Number of divisions in each direction
    element_types : dict, optional
        Output from classifyAllElementsWEB(). If None, will be computed.
    bspline_classification : dict, optional
        Output from classifyBsplinesHollig(). If None, will be computed.
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig, axes : matplotlib figure and axes (3 subplots)
    """
    # Compute classifications if not provided
    if element_types is None:
        print("[1/3] Classifying elements...")
        element_types = FEM_WEB.classifyAllElementsWEB(
            model, p, q, knotvector_x, knotvector_y, xDivision, yDivision
        )

    if bspline_classification is None:
        print("[2/3] Classifying B-splines...")
        bspline_classification = FEM_WEB.classifyBsplinesHollig(
            element_types, p, q, knotvector_x, knotvector_y, xDivision, yDivision
        )

    print("[3/3] Rendering visualization...")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # =========================================================================
    # Subplot 1: Domain boundary
    # =========================================================================
    ax = axes[0]
    n_contour = 200
    x_margin = 0.1 * (knotvector_x[-1] - knotvector_x[0])
    y_margin = 0.1 * (knotvector_y[-1] - knotvector_y[0])
    x_min = knotvector_x[0] - x_margin
    x_max = knotvector_x[-1] + x_margin
    y_min = knotvector_y[0] - y_margin
    y_max = knotvector_y[-1] + y_margin

    x_grid = np.linspace(x_min, x_max, n_contour)
    y_grid = np.linspace(y_min, y_max, n_contour)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    points = torch.tensor(
        np.column_stack([X_grid.flatten(), Y_grid.flatten()]),
        dtype=torch.float64,
    )

    with torch.no_grad():
        D = model(points).detach().cpu().numpy().reshape(X_grid.shape)

    # Plot domain: inside (d>0) and boundary (d~0)
    ax.contourf(X_grid, Y_grid, D, levels=[-1, 0], colors=["lightblue"], alpha=0.5)
    ax.contour(X_grid, Y_grid, D, levels=[0], colors=["blue"], linewidths=2)

    # Draw knot lines
    for xi in knotvector_x:
        ax.axvline(xi, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    for eta in knotvector_y:
        ax.axhline(eta, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title("Domain (SDF Level Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    # =========================================================================
    # Subplot 2: Element classification
    # =========================================================================
    ax = axes[1]

    # Plot domain boundary again
    ax.contourf(X_grid, Y_grid, D, levels=[-1, 0], colors=["lightblue"], alpha=0.3)
    ax.contour(X_grid, Y_grid, D, levels=[0], colors=["blue"], linewidths=2)

    # Draw knot lines
    for xi in knotvector_x:
        ax.axvline(xi, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    for eta in knotvector_y:
        ax.axhline(eta, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    # Draw elements with colors
    elem_colors = {"inner": "green", "boundary": "orange", "outer": "red"}
    elem_labels = {"inner": "Inner", "boundary": "Boundary", "outer": "Outer"}

    for elem_type, (color, label) in zip(
        ["inner", "boundary", "outer"], 
        [(elem_colors[k], elem_labels[k]) for k in ["inner", "boundary", "outer"]]
    ):
        for elemx, elemy, x1, x2, y1, y2 in element_types.get(elem_type, []):
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor="black", facecolor=color, alpha=0.4,
                label=label if elem_type == "inner" else None  # Only label first
            )
            ax.add_patch(rect)

    # Add legend (manually, since we only labeled the first of each type)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", alpha=0.4, label="Inner"),
        Patch(facecolor="orange", edgecolor="black", alpha=0.4, label="Boundary"),
        Patch(facecolor="red", edgecolor="black", alpha=0.4, label="Outer"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(
        f"Element Classification\n"
        f"Inner: {len(element_types.get('inner', []))} | "
        f"Boundary: {len(element_types.get('boundary', []))} | "
        f"Outer: {len(element_types.get('outer', []))}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    # =========================================================================
    # Subplot 3: B-spline classification
    # =========================================================================
    ax = axes[2]

    # Plot domain boundary again
    ax.contourf(X_grid, Y_grid, D, levels=[-1, 0], colors=["lightblue"], alpha=0.3)
    ax.contour(X_grid, Y_grid, D, levels=[0], colors=["blue"], linewidths=2)

    # Draw knot lines
    for xi in knotvector_x:
        ax.axvline(xi, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    for eta in knotvector_y:
        ax.axhline(eta, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    # Get B-spline support boxes (simplified: knot span per B-spline index)
    n_basis_x = bspline_classification["n_basis_x"]
    n_basis_y = bspline_classification["n_basis_y"]
    inner_set = set(bspline_classification["inner"])
    outer_set = set(bspline_classification["outer"])
    print("inner")
    print(inner_set)
    print("outer")
    print(outer_set)
    print("completely_outside")
    print(bspline_classification["completely_outside"])
    # Draw B-spline support regions
    for i in range(n_basis_x):
        for j in range(n_basis_y):
            # Approximate support box (simplified: based on knot spans)
            x1_approx = knotvector_x[i]
            x2_approx = knotvector_x[min(i  + 1, len(knotvector_x) - 1)]
            y1_approx = knotvector_y[j]
            y2_approx = knotvector_y[min(j + 1, len(knotvector_y) - 1)]

            if (i, j) in inner_set:
                color = "green"
                alpha = 0.3
                linewidth = 0.5
            elif (i, j) in outer_set:
                color = "orange"
                alpha = 0.3
                linewidth = 1.0
            else:
                # Completely outside domain
                color = "red"
                alpha = 0.8
                linewidth = 0.3

            rect = Rectangle(
                (x1_approx, y1_approx),
                x2_approx - x1_approx,
                y2_approx - y1_approx,
                linewidth=linewidth,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

    # Add legend
    legend_elements = [
        Patch(facecolor="green", edgecolor="green", alpha=0.3, label="Inner"),
        Patch(facecolor="orange", edgecolor="orange", alpha=0.3, label="Outer"),
        Patch(facecolor="lightgray", edgecolor="lightgray", alpha=0.1, label="Outside"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(
        f"B-spline Classification p={p}\n"
        f"Inner: {bspline_classification['n_inner']} | "
        f"Outer: {bspline_classification['n_outer']} | "
        f"Outside: {len(bspline_classification['completely_outside'])}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    # Example usage
    from Geomertry import AnaliticalDistanceCircle, AnaliticalDistanceLshape

    print("Loading geometry model...")
    model = AnaliticalDistanceLshape()

    print("Setting up mesh...")
    DIVISIONS = 8
    ORDER = 1
    DELTA = 0.005

    default = mesh.getDefaultValues(div=DIVISIONS, order=ORDER, delta=DELTA)
    x0, y0, x1, y1, xDivision, yDivision, p, q = default
    knotvector_u, knotvector_w, weights, ctrlpts = mesh.generateRectangularMesh(*default)

    print("Visualizing...")
    fig, axes = visualize_domain_and_classification(
        model, p, q, knotvector_u, knotvector_w, xDivision, yDivision
    )

    plt.show()
