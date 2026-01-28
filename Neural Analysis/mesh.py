import numpy as np
import matplotlib.pyplot as plt
import torch
import FEM

# Use float64 for better numerical accuracy
TORCH_DTYPE = torch.float64
NP_DTYPE = np.float64


USE_SIGMOID_FOR_DISTANCE = False
TRANSFORM = None  # Options: "sigmoid", "tanh", "hollig", None
#if TRANSFORM == "trapezoid": raise NameError("Trapezoid transform is not recommended, use sigmoid or tanh instead")
TANG = 1  # Used for trapezoid transform, adjust as needed
DELTA_HOLLIG = 0.2  # Controls the width of the strip for Höllig weight function
GAMMA_HOLLIG = 3  # Controls the smoothness for Höllig weight function
def generateRectangularMesh(x0, y0, x1, y1, xDivision,yDivision,p=1,q=1):
    assert x0 < x1 and y0 < y1
    knotvector_u = np.linspace(x0,x1,xDivision+2)
    knotvector_w = np.linspace(y0,y1,yDivision+2)
    weights = np.ones((yDivision+q+1,xDivision+p+1))
    ctrlpts = []
    controlx = np.linspace(x0,x1,xDivision+p+1)
    controly = np.linspace(y0,y1,yDivision+q+1)
    for j in controly:
        row = []
        for i in controlx:
            point = [i, j,1]
            row.append(point)
        ctrlpts.append(row)

    knotvector_u = np.insert(knotvector_u,0,[x0 for _ in range(p)])
    knotvector_u = np.append(knotvector_u,[x1 for _ in range(p)])
    knotvector_w = np.insert(knotvector_w,0,[y0 for _ in range(q)])
    knotvector_w = np.append(knotvector_w,[y1 for _ in range(q)])
    return knotvector_u, knotvector_w, weights, ctrlpts
def getDefaultValues(div=2,order=1,delta = 0):
    assert delta >=0
    x0 = FEM.DOMAIN["x1"] - delta
    x1 = FEM.DOMAIN["x2"] + delta
    y0 = FEM.DOMAIN["y1"] - delta
    y1 = FEM.DOMAIN["y2"] + delta
    p = order
    q = order
    xDivision = div
    yDivision = div
    return x0, y0,x1,y1,xDivision,yDivision,p,q
def distanceFromContur(x,y,model,transform=TRANSFORM):
    crd = torch.tensor([x,y],requires_grad=False,dtype=TORCH_DTYPE)
    d = model(crd)
    if transform == "sigmoid":
        d = sigmoid(d).item()
    elif transform == "tanh":
        d = torch.tanh(d).item()
    elif transform == "logarithmic":
        d = logarithmic(d).item()
    elif transform == "exponential":
        d = exponential(d).item()
    elif transform == "trapezoid":
        d = torch.where(d * TANG < 1, d * TANG, 1)
    elif transform == "hollig":
        d = hollig_weight(d).item()
    return d
def distance_with_derivative(x,y,model,transform=TRANSFORM):
    crd = torch.tensor([x,y],requires_grad=True,dtype=TORCH_DTYPE)
    d = model(crd)
    d.backward()
    dx = crd.grad[0].item()
    dy = crd.grad[1].item()
    crd.grad.zero_()
    if transform == "sigmoid":
        dx = sigmoid_derivative(d) * dx
        dy = sigmoid_derivative(d) * dy
        d = sigmoid(d).item()
    elif transform == "tanh":
        d = torch.tanh(d).item()
        dx = (1 - d**2) * dx
        dy = (1 - d**2) * dy
    elif transform == "trapezoid":
        d = torch.where(d * TANG < 1, d * TANG, 1)
        mask = d.squeeze()*TANG > 1
        dx = torch.where(mask, torch.tensor(0.0, dtype=TORCH_DTYPE), dx * TANG)
        dy = torch.where(mask, torch.tensor(0.0, dtype=TORCH_DTYPE), dy * TANG)
    elif transform == "hollig":
        dw_dd = hollig_weight_derivative(d)
        dx = dw_dd * dx
        dy = dw_dd * dy
        d = hollig_weight(d).item()
    return d,dx,dy
def distance_with_derivative_vect_trasformed(x,y,model,transform=TRANSFORM):
    crd = torch.tensor(np.array([x,y]),dtype=TORCH_DTYPE).T
    crd.requires_grad = True
    d = model(crd)
    grds = torch.autograd.grad(outputs=d, inputs=crd, grad_outputs=torch.ones_like(d),retain_graph=True)
    dx = grds[0][:,0]
    dy = grds[0][:,1]
    if transform == "sigmoid":
        dx = sigmoid_derivative(d).view(-1) * dx
        dy = sigmoid_derivative(d).view(-1) * dy
        d = sigmoid(d)
    elif transform == "tanh":
        d = torch.tanh(d)
        dx = (1 - d**2).view(-1) * dx
        dy = (1 - d**2).view(-1) * dy
    elif transform == "logarithmic":
        dx = (1 / (d + 1)).view(-1) * dx
        dy = (1 / (d + 1)).view(-1) * dy
        d = logarithmic(d)
    elif transform == "exponential":
        dx = torch.exp(d).view(-1) * dx
        dy = torch.exp(d).view(-1) * dy
        d = exponential(d)
    elif transform == 'trapezoid':
        
        d = torch.where(d * TANG < 1, d * TANG, 1)
        mask = d.squeeze()*TANG > 1
        dx = torch.where(mask, torch.tensor(0.0, dtype=TORCH_DTYPE), dx * TANG)
        dy = torch.where(mask, torch.tensor(0.0, dtype=TORCH_DTYPE), dy * TANG)
    elif transform == "hollig":
        dw_dd = hollig_weight_derivative(d).view(-1)
        dx = dw_dd * dx
        dy = dw_dd * dy
        d = hollig_weight(d)

    #crd.grad.zero_()
    return d,dx,dy
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    tmp = torch.ones_like(s) - s
    tmp2 = s * (torch.ones_like(s) - s)
    return tmp2
def logarithmic(x):
    return torch.log(x + 1)
def exponential(x):
    return torch.exp(x) - 1
def hollig_weight(d, delta=DELTA_HOLLIG, gamma=GAMMA_HOLLIG):
    """Höllig weight function: w(x) = 1 - max(0, 1 - dist/delta)^gamma"""
    term = 1.0 - d / delta
    term = torch.clamp(term, min=0.0)  # max(0, 1 - d/delta)
    return 1.0 - torch.pow(term, gamma)
def hollig_weight_derivative(d, delta=DELTA_HOLLIG, gamma=GAMMA_HOLLIG):
    """Derivative of Höllig weight function with respect to distance d"""
    term = 1.0 - d / delta
    mask = term > 0  # Only non-zero derivative where term > 0
    derivative = torch.zeros_like(d)
    derivative[mask] = gamma * torch.pow(term[mask], gamma - 1) / delta
    return derivative
def plotMesh(xdiv=2, ydiv=3,delta=0):
    circle = plt.Circle((0, 0), 1, color='r')
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    ax.set_xticks(np.arange(-1-delta, 1+delta, (2+2*delta)/(xdiv+1)))
    ax.set_yticks(np.arange(-1-delta, 1+delta, (2+2*delta)/(ydiv+1)))
    ax.set_xlim((-1-delta,1+delta))
    ax.set_ylim((-1-delta,1+delta))
    ax.set_aspect('equal')
    
    plt.grid(True)
    plt.show()
def plotDisctancefunction(model):
    N = 200
    x_values = np.linspace(FEM.DOMAIN["x1"]-0.1, FEM.DOMAIN["x2"]+0.1, N)
    y_values = np.linspace(FEM.DOMAIN["y1"]-0.1, FEM.DOMAIN["y2"]+0.1, N)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros((N,N))
    # Evaluate the function at each point in the grid
    for idxx, xx in enumerate(x_values):
        for idxy,yy in enumerate(y_values):
            Z[idxy, idxx] = distanceFromContur(xx,yy,model)

    #Z = distanceFromContur(X, Y)

    # Create a contour plot
    plt.contourf(X, Y, Z,levels=20)
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scalar-Valued Function f(x, y)')
    plt.grid(True)
    highlight_level = 0.0
    plt.contour(X, Y, Z, levels=[highlight_level], colors='red')
    plt.show()
def plotAlayticHeatmap(solfun,n=10):
    x_values = np.linspace(-1.1, 1.1, 1000)
    y_values = np.linspace(-1.1, 1.1, 1000)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros((1000,1000))
    # Evaluate the function at each point in the grid
    for idxx, xx in enumerate(x_values):
        for idxy,yy in enumerate(y_values):
            Z[idxy, idxx] = solfun(xx,yy)

    #Z = distanceFromContur(X, Y)

    # Create a contour plot
    plt.contourf(X, Y, Z,levels=100)
    plt.colorbar(label='u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('A modellprobléma megoldása')
    plt.grid(False)
    plt.show()
if __name__ == "__main__":
    import Geomertry
    model = Geomertry.AnaliticalDistanceCircle()
    #Create a 3D plot of the SDF
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import torch
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-1.01, 1.01, 100)
    y = np.linspace(-1.01, 1.01, 100)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float64)
    #target = SDF.distance_from_line_vectorized(pts, angle=math.pi/6, offset=0.0)

    pred = distance_with_derivative_vect_trasformed(X.ravel(), Y.ravel(), model, transform=TRANSFORM)[0].detach().numpy()

    Z = pred.reshape(X.shape)
    Z = np.clip(Z,0, 2)
    print(np.max(Z))
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()
