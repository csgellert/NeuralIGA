import numpy as np
import torch
import NeuralImplicit as NI
import matplotlib.pyplot as plt
import geometry_bspline as bsp_geom

def generate_2d_ground_truth(div=100,eps = 0.1):
    x = np.linspace(-1-eps, 1+eps, div)
    y = np.linspace(-1-eps, 1+eps, div)
    X, Y = np.meshgrid(x, y)
    bsp_cp = bsp_geom.create_star_bspline_control_points(center=(0, 0), outer_radius=1.0, inner_radius=0.5, num_star_points=5, degree=1)
    Z = bsp_geom.bspline_signed_distance_vectorized(torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=-1), dtype=torch.float32), bsp_cp)
    Z = Z.view(div, div).cpu().numpy()
    #save data for later use
    print("Saving star shape data to star_shape.npz")
    np.savez('star_shape.npz',X=X,Y=Y,Z=Z)

if __name__ == "__main__":
    generate_2d_ground_truth(div=100,eps=0)
    
