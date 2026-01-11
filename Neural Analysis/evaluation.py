import FEM
from bspline import Bspline
import numpy as np
import torch
import Geomertry
from matplotlib import pyplot as plt
import mesh
import json
from pathlib import Path

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)
TORCH_DTYPE = torch.float64


def evaluateAccuracy(model, results, p, q, knotvector_x, knotvector_y, N=1000, seed=None):
    """
    Evaluate the accuracy of the numerical solution compared to the analytical solution.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model representing the geometry (SDF)
    results : array-like
        Coefficients of the B-spline solution
    p : int
        B-spline degree in x direction
    q : int
        B-spline degree in y direction
    knotvector_x : array-like
        Knot vector in x direction
    knotvector_y : array-like
        Knot vector in y direction
    N : int
        Number of random evaluation points
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing all error metrics:
        - 'MSE': Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'L_inf': Maximum absolute error (L-infinity norm)
        - 'relative_error': Relative L2 error
        - 'H1_error': H1 seminorm error (gradient error)
        - 'H1_full': Full H1 norm error (includes L2 and gradient)
        - 'n_valid_points': Number of points inside the domain
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points in the domain
    x_samples = np.random.uniform(FEM.DOMAIN["x1"], FEM.DOMAIN["x2"], N)
    y_samples = np.random.uniform(FEM.DOMAIN["y1"], FEM.DOMAIN["y2"], N)
    
    # Create Bspline objects for evaluation
    Bspxi = Bspline(knotvector_x, p)
    Bspeta = Bspline(knotvector_y, q)
    dBspxi = Bspxi.diff(1)  # First derivative
    dBspeta = Bspeta.diff(1)
    
    # Number of basis functions
    n_basis_x = len(knotvector_x) - p - 1
    n_basis_y = len(knotvector_y) - q - 1
    
    # Lists to store results for valid points (inside domain)
    numerical_vals = []
    analytical_vals = []
    numerical_dx = []
    numerical_dy = []
    analytical_dx = []
    analytical_dy = []
    
    # Evaluate at each point
    points_tensor = torch.tensor(np.column_stack([x_samples, y_samples]), dtype=TORCH_DTYPE)
    
    with torch.no_grad():
        distances = model(points_tensor).numpy().flatten()
    
    for idx in range(N):
        xx, yy = x_samples[idx], y_samples[idx]
        d = distances[idx]
        
        # Skip points outside the domain
        if d < 0:
            continue
        
        # Get distance and its derivatives for points inside
        d_tensor, dx_d, dy_d = mesh.distance_with_derivative(xx, yy, model)
        if hasattr(d_tensor, 'item'):
            d_val = d_tensor.item() if hasattr(d_tensor, 'item') else float(d_tensor)
        else:
            d_val = float(d_tensor)
        if hasattr(dx_d, 'item'):
            dx_d = dx_d.item() if hasattr(dx_d, 'item') else float(dx_d)
        if hasattr(dy_d, 'item'):
            dy_d = dy_d.item() if hasattr(dy_d, 'item') else float(dy_d)
        
        # Compute numerical solution: u_h = d * sum(N_i * c_i) + (1-d) * g
        # where g is the Dirichlet boundary condition
        
        # Evaluate B-spline basis functions using Bspline objects
        bxi = Bspxi(xx)  # Returns array of all basis function values at xx
        beta = Bspeta(yy)  # Returns array of all basis function values at yy
        dbxi = dBspxi(xx)  # Derivative values
        dbeta = dBspeta(yy)  # Derivative values
        
        # Compute the B-spline approximation sum(N_i * c_i)
        bspline_sum = 0.0
        bspline_sum_dx = 0.0
        bspline_sum_dy = 0.0
        
        for i in range(n_basis_x):
            for j in range(n_basis_y):
                coeff = results[n_basis_x * i + j]
                N_ij = bxi[i] * beta[j]
                dN_ij_dx = dbxi[i] * beta[j]
                dN_ij_dy = bxi[i] * dbeta[j]
                
                bspline_sum += N_ij * coeff
                bspline_sum_dx += dN_ij_dx * coeff
                bspline_sum_dy += dN_ij_dy * coeff
        
        # Get Dirichlet boundary values and derivatives
        g = FEM.dirichletBoundary(xx, yy)
        dg_dx = FEM.dirichletBoundaryDerivativeX(xx, yy)
        dg_dy = FEM.dirichletBoundaryDerivativeY(xx, yy)
        
        # Numerical solution: u_h = d * phi + (1-d) * g
        u_h = d_val * bspline_sum + (1 - d_val) * g
        
        # Gradient of numerical solution using product rule:
        # du_h/dx = d_x * phi + d * phi_x + (-d_x) * g + (1-d) * g_x
        #         = d_x * (phi - g) + d * phi_x + (1-d) * g_x
        du_h_dx = dx_d * (bspline_sum - g) + d_val * bspline_sum_dx + (1 - d_val) * dg_dx
        du_h_dy = dy_d * (bspline_sum - g) + d_val * bspline_sum_dy + (1 - d_val) * dg_dy
        
        # Analytical solution and derivatives
        u_exact = FEM.solution_function(xx, yy)
        du_exact_dx = FEM.solution_function_derivative_x(xx, yy)
        du_exact_dy = FEM.solution_function_derivative_y(xx, yy)
        
        # Store values
        numerical_vals.append(u_h)
        analytical_vals.append(u_exact)
        numerical_dx.append(du_h_dx)
        numerical_dy.append(du_h_dy)
        analytical_dx.append(du_exact_dx)
        analytical_dy.append(du_exact_dy)
    
    # Convert to numpy arrays
    numerical_vals = np.array(numerical_vals)
    analytical_vals = np.array(analytical_vals)
    numerical_dx = np.array(numerical_dx)
    numerical_dy = np.array(numerical_dy)
    analytical_dx = np.array(analytical_dx)
    analytical_dy = np.array(analytical_dy)
    
    n_valid = len(numerical_vals)
    
    if n_valid == 0:
        return {
            'MSE': np.nan,
            'MAE': np.nan,
            'L_inf': np.nan,
            'relative_error': np.nan,
            'H1_error': np.nan,
            'H1_full': np.nan,
            'n_valid_points': 0
        }
    
    # Compute errors
    error = numerical_vals - analytical_vals
    error_dx = numerical_dx - analytical_dx
    error_dy = numerical_dy - analytical_dy
    
    # MSE: Mean Squared Error
    MSE = np.mean(error ** 2)
    
    # MAE: Mean Absolute Error
    MAE = np.mean(np.abs(error))
    
    # L_inf: Maximum absolute error
    L_inf = np.max(np.abs(error))
    
    # Relative MAE: MAE / mean(|u|)
    MAE_exact = np.mean(np.abs(analytical_vals))
    relative_error = MAE / MAE_exact if MAE_exact > 1e-15 else np.inf
    
    # H1 seminorm error: ||grad(u_h - u)||_2
    H1_seminorm_sq = np.sum(error_dx ** 2 + error_dy ** 2)
    H1_error = np.sqrt(H1_seminorm_sq / n_valid)
    
    # Full H1 norm error: sqrt(||u_h - u||_2^2 + ||grad(u_h - u)||_2^2)
    H1_full = np.sqrt(np.sum(error ** 2) + H1_seminorm_sq) / np.sqrt(n_valid)
    
    results_dict = {
        'MSE': MSE,
        'MAE': MAE,
        'L_inf': L_inf,
        'relative_error': relative_error,
        'H1_error': H1_error,
        'H1_full': H1_full,
        'n_valid_points': n_valid
    }
    
    return results_dict


def printErrorMetrics(metrics):
    """Pretty print the error metrics."""
    print("=" * 50)
    print("Error Metrics Summary")
    print("=" * 50)
    print(f"  Valid evaluation points: {metrics['n_valid_points']}")
    print(f"  MSE (Mean Squared Error):    {metrics['MSE']:.6e}")
    print(f"  MAE (Mean Absolute Error):   {metrics['MAE']:.6e}")
    print(f"  L_inf (Max Absolute Error):  {metrics['L_inf']:.6e}")
    print(f"  Relative Absolute Error:     {metrics['relative_error']:.6e}")
    print(f"  H1 Seminorm Error:           {metrics['H1_error']:.6e}")
    print(f"  H1 Full Norm Error:          {metrics['H1_full']:.6e}")
    print("=" * 50)
def visualizeResultsBspline(model,results,p,q,knotvector_x, knotvector_y):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(FEM.DOMAIN["x1"], FEM.DOMAIN["x2"], 40)
    y = np.linspace(FEM.DOMAIN["y1"], FEM.DOMAIN["y2"], 40)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model)
            analitical.append(FEM.solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += Geomertry.B(xx,p,xbasis,knotvector_x)*Geomertry.B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*FEM.dirichletBoundary(xx,yy)
            if d<0: sum = 0
            try:
                result.append(sum.item())
            except:
                result.append(sum)
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(xPoints,yPoints,result)#, edgecolors='face')
    ax.scatter(xPoints,yPoints,analitical)

    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    MAE = np.abs(np.array(result)-np.array(analitical)).mean()
    print(f"MAE: {MAE}")
    L_inf_error = np.max(np.abs(np.array(result)-np.array(analitical)))
    print(f"L_inf error: {L_inf_error}")
    plt.show()
def calculateErrorBspline(model,results,p,q,knotvector_x, knotvector_y,larger_domain=True):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    x = np.linspace(FEM.DOMAIN["x1"], FEM.DOMAIN["x2"], 40)
    y = np.linspace(FEM.DOMAIN["y1"], FEM.DOMAIN["y2"], 40)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            analitical.append(FEM.solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += Geomertry.B(xx,p,xbasis,knotvector_x)*Geomertry.B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*FEM.dirichletBoundary(xx,yy)
            if d<0: sum = 0
            result.append(sum)
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    return MSE
def plotErrorHeatmap(model,results,knotvector_x,knotvector_y,p,q,larger_domain = True,N=150):

    marg = 0.1
    x = np.linspace(FEM.DOMAIN["x1"]-marg,FEM.DOMAIN["x2"]+marg,N)
    y = np.linspace(FEM.DOMAIN["y1"]-marg,FEM.DOMAIN["y2"]+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_N=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += Geomertry.B(xx,p,xbasis,knotvector_x)*Geomertry.B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*FEM.dirichletBoundary(xx,yy)
            if d<0: sum = 0
            Z_N[idxx,idxy] = sum
    #MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    #print(f"MSE: {MSE}")
    #return MSE
    plt.contourf(X, Y, Z_N,levels=20)
    points = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0),(1.0,0.5),(0.5,0.5),(0.5,1.0),(0.0, 1.0)]

    # Plot red lines between the points
    for i in range(len(points)-1):
            plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'r-')

    #plt.axis('equal')
    #highlight_level = 0.0
    #plt.contour(X, Y, Z, levels=[highlight_level], colors='red')
    plt.colorbar()
    plt.title('Solution of the PDE')
    plt.grid(True)
    plt.show()
def plotResultHeatmap(model,results,knotvector_x,knotvector_y,p,q,larger_domain = True,N=150):
    marg = 0.05
    x = np.linspace(FEM.DOMAIN["x1"]-marg,FEM.DOMAIN["x2"]+marg,N)
    y = np.linspace(FEM.DOMAIN["y1"]-marg,FEM.DOMAIN["y2"]+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_A=np.zeros((N,N))
    Z_N=np.zeros((N,N))
    ERR=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model)
            Z_A[idxx,idxy] = FEM.solution_function(xx,yy) if d>=0 else 0
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += Geomertry.B(xx,p,xbasis,knotvector_x)*Geomertry.B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*FEM.dirichletBoundary(xx,yy)
            if d<0: sum = 0
            Z_N[idxx,idxy] = sum
    #MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    #print(f"MSE: {MSE}")
    #return MSE
    ERR = np.abs(Z_N-Z_A)
    plt.contourf(X, Y, Z_A,levels=20)
    #plt.axis('equal')
    plt.colorbar()
    highlight_level = 0.0
    plt.contour(X, Y, Z_A, levels=[highlight_level], colors='red') 

    # Show the plot
    
    plt.title('Solution')
    plt.grid(True)
    plt.show()


#* TEST
if __name__ == "__main__":
    from NeuralImplicit import Siren
    siren_model_kor_jo = Siren(in_features=2,out_features=1,hidden_features=256,hidden_layers=2,outermost_linear=True)
    siren_model_kor_jo.load_state_dict(torch.load('siren_model_kor_jo.pth',weights_only=True,map_location=torch.device('cpu')))
    siren_model_kor_jo.eval()
    model = siren_model_kor_jo
    test_values = [20,30,40,50,60,80,120]
    esize = [1/(nd+1) for nd in test_values]
    orders = [2]
    fig,ax = plt.subplots()
    for order in orders:
        accuracy = []
        etypes = []
        for division in test_values:
            etype = {"outer":0,"inner":0,"boundary":0}
            default = mesh.getDefaultValues(div=division,order=order,delta=0.005)
            x0, y0,x1,y1,xDivision,yDivision,p,q = default
            knotvector_u, knotvector_w,weigths, ctrlpts = mesh.generateRectangularMesh(*default)
            assert p==q and xDivision == yDivision
            for elemx in range(p,p+xDivision+1):
                for elemy in range(q,q+xDivision+1):
                    etype = FEM.elementTypeChoose(knotvector_u,knotvector_w,elemx,elemy,etype)
            etypes.append(etype)
    print(etypes)


def load_simulation_results(filepath):
    """Load simulation results from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def visualize_results(filepath, metric='MAE', log_scale=True):
    """
    Visualize simulation results from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON results file
    metric : str
        Error metric to visualize. Options: 'MSE', 'MAE', 'L_inf', 
        'relative_error', 'H1_error', 'H1_full'
    log_scale : bool
        Whether to use logarithmic scale for y-axis
    """
    valid_metrics = ['MSE', 'MAE', 'L_inf', 'relative_error', 'H1_error', 'H1_full']
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {valid_metrics}")
    
    # Load data
    data = load_simulation_results(filepath)
    
    model_name = data['model_name']
    test_values = data['test_values']
    orders = data['orders']
    eval_stats = data['eval_stats']
    timestamp = data['timestamp']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each order
    markers = ['o', 's', '^', 'd', 'v', '<', '>']
    colors = plt.cm.tab10.colors
    
    for idx, order in enumerate(orders):
        metric_values = []
        divisions = []
        
        for div in test_values:
            key = f"order_{order}_div_{div}"
            if key in eval_stats:
                metric_values.append(eval_stats[key][metric])
                divisions.append(div)
        
        if divisions:
            # Element size = domain_size / (divisions + 1)
            element_sizes = [1.0 / (d + 1) for d in divisions]
            
            ax.plot(element_sizes, metric_values, 
                   marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)],
                   label=f'p = {order}',
                   linewidth=2, markersize=8)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('Element Size (h)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Convergence of {metric}\nModel: {model_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nResults from: {filepath}")
    print(f"Timestamp: {timestamp}")
    print(f"\n{metric} values:")
    print("-" * 50)
    for order in orders:
        print(f"\nOrder p = {order}:")
        for div in test_values:
            key = f"order_{order}_div_{div}"
            if key in eval_stats:
                print(f"  div={div:3d} (h={1/(div+1):.4f}): {eval_stats[key][metric]:.6e}")


def compare_simulations(filepaths, orders_to_show, metric='MAE', log_scale=True, 
                        labels=None, figsize=(12, 7)):
    """
    Compare simulation results from multiple JSON files on the same plot.
    
    Parameters:
    -----------
    filepaths : list of str
        List of paths to JSON results files
    orders_to_show : list of int
        List of B-spline orders to display (e.g., [1, 2, 3])
    metric : str
        Error metric to visualize. Options: 'MSE', 'MAE', 'L_inf', 
        'relative_error', 'H1_error', 'H1_full'
    log_scale : bool
        Whether to use logarithmic scale for axes
    labels : list of str, optional
        Custom labels for each file. If None, uses model names from files
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    valid_metrics = ['MSE', 'MAE', 'L_inf', 'relative_error', 'H1_error', 'H1_full']
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {valid_metrics}")
    
    if labels is not None and len(labels) != len(filepaths):
        raise ValueError("Length of labels must match length of filepaths")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define markers and line styles for different files and orders
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h', '*']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    # Track all plotted data for legend
    plot_idx = 0
    
    for file_idx, filepath in enumerate(filepaths):
        # Load data
        data = load_simulation_results(filepath)
        
        model_name = data['model_name']
        test_values = data['test_values']
        available_orders = data['orders']
        eval_stats = data['eval_stats']
        
        # Use custom label or model name
        file_label = labels[file_idx] if labels else model_name
        
        # Plot for each requested order that exists in this file
        for order in orders_to_show:
            if order not in available_orders:
                continue
                
            metric_values = []
            divisions = []
            
            for div in test_values:
                key = f"order_{order}_div_{div}"
                if key in eval_stats:
                    metric_values.append(eval_stats[key][metric])
                    divisions.append(div)
            
            if divisions:
                # Element size = domain_size / (divisions + 1)
                element_sizes = [1.0 / (d + 1) for d in divisions]
                
                # Choose style based on file and order
                marker = markers[plot_idx % len(markers)]
                linestyle = linestyles[file_idx % len(linestyles)]
                color = colors[plot_idx % len(colors)]
                
                ax.plot(element_sizes, metric_values, 
                       marker='*',
                       linestyle='-',
                       color=color,
                       label=f'{file_label}, p={order}',
                       linewidth=2, markersize=8)
                
                plot_idx += 1
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('Element Size (h)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Comparison of {metric} Convergence', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 70)
    print(f"Comparison Summary - Metric: {metric}")
    print("=" * 70)
    
    for file_idx, filepath in enumerate(filepaths):
        data = load_simulation_results(filepath)
        file_label = labels[file_idx] if labels else data['model_name']
        print(f"\n{file_label}:")
        print("-" * 50)
        
        for order in orders_to_show:
            if order not in data['orders']:
                continue
            print(f"  Order p = {order}:")
            for div in data['test_values']:
                key = f"order_{order}_div_{div}"
                if key in data['eval_stats']:
                    val = data['eval_stats'][key][metric]
                    print(f"    div={div:3d} (h={1/(div+1):.4f}): {val:.6e}")
    
    return fig, ax





