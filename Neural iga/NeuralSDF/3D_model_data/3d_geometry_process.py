import numpy as np
import torch
import NeuralImplicit
import matplotlib.pyplot as plt

def open_object(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(part.split('/')[0]) - 1 for part in parts[1:4]]
                faces.append(face)
    return np.array(vertices), np.array(faces)

def create_torch_mesh(vertices, faces, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(faces, dtype=torch.int64, device=device)
    return vertices_tensor, faces_tensor

def get_signed_distance_from_contour(vertices, faces, points):
    """
    Compute signed distance from points to a 3D mesh.
    Positive distances are inside the mesh, negative distances are outside.
    Optimized for runtime performance.
    """
    device = points.device
    batch_size = points.shape[0]
    
    # Compute unsigned distance (closest distance to surface vertices)
    # This is faster than computing distance to triangle faces for large meshes
    d = torch.cdist(points, vertices)
    min_distances, _ = torch.min(d, dim=1)
    
    # Optimized ray casting with precomputed triangle data
    ray_direction = torch.tensor([1.0, 0.0, 0.0], device=device)
    signs = torch.ones(batch_size, device=device)
    
    # Precompute triangle data once
    face_vertices = vertices[faces]  # Shape: [num_faces, 3, 3]
    v0 = face_vertices[:, 0]  # [num_faces, 3]
    v1 = face_vertices[:, 1]  # [num_faces, 3]
    v2 = face_vertices[:, 2]  # [num_faces, 3]
    
    # Precompute edge vectors
    edges1 = v1 - v0  # [num_faces, 3]
    edges2 = v2 - v0  # [num_faces, 3]
    
    # Precompute h vectors for all faces
    h_vectors = torch.cross(ray_direction.unsqueeze(0).expand(len(faces), -1), edges2, dim=1)
    determinants = torch.sum(edges1 * h_vectors, dim=1)
    
    # Filter out nearly parallel triangles once
    valid_mask = torch.abs(determinants) > 1e-8
    valid_faces_indices = torch.where(valid_mask)[0]
    
    if len(valid_faces_indices) == 0:
        # All faces are parallel to ray, assume all points are outside
        return -min_distances
    
    # Extract valid face data
    valid_v0 = v0[valid_faces_indices]
    valid_edges1 = edges1[valid_faces_indices]
    valid_edges2 = edges2[valid_faces_indices]
    valid_h = h_vectors[valid_faces_indices]
    valid_det = determinants[valid_faces_indices]
    inv_det = 1.0 / valid_det
    
    # Process points in batches to balance memory and speed
    batch_size_chunk = min(1000, batch_size)
    
    for start_idx in range(0, batch_size, batch_size_chunk):
        end_idx = min(start_idx + batch_size_chunk, batch_size)
        chunk_points = points[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        # Vectorized intersection computation for this chunk
        intersection_counts = torch.zeros(chunk_size, device=device, dtype=torch.int32)
        
        for i, point in enumerate(chunk_points):
            # Compute s vectors for all valid faces
            s_vectors = point.unsqueeze(0) - valid_v0  # [num_valid_faces, 3]
            
            # Compute u coordinates
            u_coords = inv_det * torch.sum(s_vectors * valid_h, dim=1)
            
            # Early filtering for u coordinate bounds
            u_valid_mask = (u_coords >= 0.0) & (u_coords <= 1.0)
            u_valid_indices = torch.where(u_valid_mask)[0]
            
            if len(u_valid_indices) > 0:
                # Compute v coordinates only for valid u
                q_vectors = torch.cross(s_vectors[u_valid_indices], valid_edges1[u_valid_indices], dim=1)
                v_coords = inv_det[u_valid_indices] * torch.sum(ray_direction.unsqueeze(0) * q_vectors, dim=1)
                
                # Check v bounds and triangle constraint
                v_valid_mask = (v_coords >= 0.0) & (u_coords[u_valid_indices] + v_coords <= 1.0)
                v_valid_indices = u_valid_indices[v_valid_mask]
                
                if len(v_valid_indices) > 0:
                    # Compute intersection parameter t
                    final_q = q_vectors[v_valid_mask]
                    t_coords = inv_det[v_valid_indices] * torch.sum(valid_edges2[v_valid_indices] * final_q, dim=1)
                    
                    # Count intersections in front of ray
                    intersection_counts[i] = torch.sum(t_coords > 1e-8).item()
        
        # Determine signs for this chunk
        inside_mask = (intersection_counts % 2) == 1
        signs[start_idx:end_idx] = torch.where(inside_mask, 1.0, -1.0)
    
    return signs * min_distances
def train_model():
    model = NeuralImplicit.Siren(3, 256, 5, 1, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.)
    optimizer = torch.optim.Adam(lr=1e-6, params=model.parameters())
    geom_model = open_object("3D_model_data/stanford-bunny.obj")
    vertices, faces = geom_model
    vertices_tensor, faces_tensor = create_torch_mesh(vertices, faces)
    print("Model loaded successfully")
    loss_history = []
    for epoch in range(500):
        optimizer.zero_grad()
        # Dummy input and target for illustration purposes
        input_points = (torch.rand((10000, 3))-0.5)*0.2
        target_distances = get_signed_distance_from_contour(vertices_tensor, faces_tensor, input_points)
        predicted_distances = model(input_points)
        loss = torch.nn.functional.mse_loss(predicted_distances, target_distances)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss_history.append(loss.item())
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss Over Time')
    plt.show()

if __name__ == "__main__":
    geom_model = open_object("3D_model_data/stanford-bunny.obj")
    vertices, faces = geom_model
    vertices_tensor, faces_tensor = create_torch_mesh(vertices, faces)
    print("Model loaded successfully")
    test_points = torch.tensor([[0.0, 0.0, 0.0],
                                [-0.037830, 0.126940, 0.004475],
                                [0.2, 0.2, 0.5]], dtype=torch.float32)
                            
    distances = get_signed_distance_from_contour(vertices_tensor, faces_tensor, test_points)
    print("Distances from contour:", distances)
    train_model()
    """
    print("max x value:", np.max(vertices[:,0]))
    print("min x value:", np.min(vertices[:,0]))
    print("max y value:", np.max(vertices[:,1])) 
    print("min y value:", np.min(vertices[:,1]))
    print("max z value:", np.max(vertices[:,2]))
    print("min z value:", np.min(vertices[:,2]))
    print("first vertex:", vertices_tensor[0])
    tmp = torch.cdist(test_points, vertices_tensor[0:2])
    print(tmp)
    min_distances, _ = torch.min(tmp, dim=1)
    print("Distances from contour (manual):", min_distances)
    
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)
    ax.set_title('3D Scatter Plot of Stanford Bunny Vertices')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    """