import FEM
from bspline import BSpline
import numpy as np
import torch
import Geomertry
from matplotlib import pyplot as plt
import mesh

# Use float64 for better numerical accuracy
torch.set_default_dtype(torch.float64)
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
