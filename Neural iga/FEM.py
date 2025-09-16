from Geomertry import *
import numpy as np
import matplotlib.pyplot as plt
import mesh
import math
import torch
from bspline import Bspline

FUNCTION_CASE = 2
LARGER_DOMAIN = FUNCTION_CASE <=4 # if True, the domain is [-1,1]x[-1,1], otherwise [0,1]x[0,1]
print(f"Larger domain: {LARGER_DOMAIN}")
MAX_SUBDIVISION = 4
def load_function(x,y):
    #! -f(x)
    if FUNCTION_CASE == 1:
        return -8*x
    elif FUNCTION_CASE == 2:
        arg = (x**2 + y**2)*math.pi/2
        return -(-2*math.pi*math.sin(arg)-math.cos(arg)*(x**2 + y**2)*math.pi**2)
    elif FUNCTION_CASE == 3:
        return -8*x
    elif FUNCTION_CASE == 4:
        return -8*x
    elif FUNCTION_CASE == 5:#L-shape
        return 8*math.pi*math.pi*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
    elif FUNCTION_CASE == 6: #tube
        return -(x**2 + y**2)
    else:
        raise NotImplementedError
def solution_function(x,y):
    if FUNCTION_CASE == 1:
        return x*(x**2 + y**2 -1)
    elif FUNCTION_CASE == 2:
        return math.cos((x**2 + y**2)*math.pi/2)
    elif FUNCTION_CASE == 3:
        return x*(x**2 + y**2 -1) + 2
    elif FUNCTION_CASE == 4:
        return x*(x**2 + y**2 -1) + x +2*y
    elif FUNCTION_CASE == 5: #L-shape
        return math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
    else: raise NotImplementedError
def dirichletBoundary(x,y):
    if FUNCTION_CASE == 1:
        return 2
    if FUNCTION_CASE == 2:
        return 0
    if FUNCTION_CASE == 3:
        return 2
    if FUNCTION_CASE == 4:
        return x+2*y
    elif FUNCTION_CASE == 5:
        return 0
    else: raise NotImplementedError
def dirichletBoundaryDerivativeX(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 1
    elif FUNCTION_CASE == 5:#L-shape
        return 0
    else: raise NotImplementedError
def dirichletBoundaryDerivativeY(x,y):
    if FUNCTION_CASE <= 3:
        return 0
    elif FUNCTION_CASE == 4:
        return 2
    elif FUNCTION_CASE ==5:  #L-shape
        return 0
    else: raise NotImplementedError

def element(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
    assert q==p
    SUBDIVISION = 1
    DOSUBDIV = True
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    #doing subdivision
    if DOSUBDIV:
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=0)
    return K,F
def boundaryElement(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta):
    assert q==p
    SUBDIVISION = 1
    DOSUBDIV = True
    #* Defining Gauss points
    g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
    w = np.array([1,1])
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    #doing subdivision
    if DOSUBDIV:
        K,F = Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level=0,MAXLEVEL=MAX_SUBDIVISION)
    return K,F
def Subdivide(model,x1,x2,y1,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    halfy = (y1+y2)/2
    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    if level == MAXLEVEL:
        #first
        Ks,Fs = GaussQuadrature(model,x1, halfx,y1,halfy,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #second
        Ks,Fs = GaussQuadrature(model,halfx, x2,y1,halfy,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #third
        Ks,Fs = GaussQuadrature(model,x1, halfx,halfy,y2,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        #fourth
        Ks,Fs = GaussQuadrature(model,halfx, x2,halfy,y2,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta)
        K+=Ks/4
        F+=Fs/4
        return K,F
    else:
        Kret,Fret=Subdivide(model,x1,halfx,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,x1,halfx,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,y1,halfy,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        Kret,Fret=Subdivide(model,halfx,x2,halfy,y2,i,j,knotvector_x,knotvector_y,p,q,Bspxi,Bspeta,level+1,MAXLEVEL)
        K+= Kret/4
        F+=Fret/4
        return K,F

def GaussQuadrature(model,x1,x2,y1,y2,i,j,p,q,knotvector_x,knotvector_y,Bspxi,Bspeta):
    if p <=2:
        g = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
        w = np.array([1,1])
        gaussP_x = np.array([g[0],g[0],g[1],g[1]])
        gaussP_y = np.array([g[0],g[1],g[0],g[1]])
        gauss_weights = np.array([w[0],w[0],w[1],w[1]])
        num_gauss_points = 4
    else:
        g = np.array([-math.sqrt(3/5), 0, math.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
        gaussP_x = np.array([g[0],g[0],g[0],g[1],g[1],g[1],g[2],g[2],g[2]])
        gaussP_y = np.array([g[0],g[1],g[2],g[0],g[1],g[2],g[0],g[1],g[2]])
        gauss_weights = np.array([w[0]*w[0],w[1]*w[0],w[2]*w[0],w[0]*w[1],w[1]*w[1],w[2]*w[1],w[0]*w[2],w[1]*w[2],w[2]*w[2]])
        num_gauss_points = 9

    K = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
    F = np.zeros((p+1)*(q+1))
    Bspxi = Bspline(knotvector_x,p)
    Bspeta = Bspline(knotvector_y,q)
    
    xi = (x2-x1)/2 * gaussP_x + (x2+x1)/2
    eta = (y2-y1)/2 * gaussP_y + (y2+y1)/2

    d_,dx_,dy_ = mesh.distance_with_derivative_vect_trasformed(xi,eta,model)
    bxi_ = Bspxi.collmat(xi)
    beta_ = Bspeta.collmat(eta)
    dbdxi_ = Bspxi.collmat(xi,1)
    dbdeta_ = Bspeta.collmat(eta,1)
    for idx in range(num_gauss_points): #iterate throug Gauss points functions in x
            #d = mesh.distanceFromContur(xi,eta,model)
            d = d_[idx].item()
            if d<0: continue
            dx = dx_[idx].item()
            dy = dy_[idx].item()
            bxi = bxi_[idx]
            beta = beta_[idx]
            dbdxi = dbdxi_[idx]
            dbdeta = dbdeta_[idx]
            #*CAlculating the Ke
            for xbasisi in range(i-p,i+1):
                for ybasisi in range(j-q,j+1): 
                    dNidxi = dbdxi[xbasisi]*beta[ybasisi]
                    dNidEta = bxi[xbasisi]*dbdeta[ybasisi]
                    Ni = beta[ybasisi]*bxi[xbasisi]
                    diCorrXi = dNidxi*d + dx * Ni
                    diCorrEta = dNidEta*d + dy * Ni
                    for xbasisj in range(i-p,i+1):
                        for ybasisj in range(j-q,j+1): 
                            #f = R2(nControlx,nControly,xbasis,ybasis,xi,eta,weigths,knotvector_x,knotvector_y,p,q)
                            dNjdxi = dbdxi[xbasisj]*beta[ybasisj]
                            dNjdeta = bxi[xbasisj]*dbdeta[ybasisj]
                            Nj = beta[ybasisj]*bxi[xbasisj]
                            #correction with the distance function
                            djCorrXi = dNjdxi*d + dx * Nj
                            djCorrEta = dNjdeta*d + dy * Nj
                            #Not anymore: In the line below the weigths have been taken out
                            K[(p+1)*(xbasisi-(i-p)) + ybasisi-(j-q)][(p+1)*(xbasisj-(i-p))+(ybasisj-(j-q))] += (((diCorrXi)*(djCorrXi) + (diCorrEta)*(djCorrEta)))*gauss_weights[idx]
                    fi = load_function(xi[idx], eta[idx])
                    Ni_corr = d*Ni
                    dirichlet_xi_eta = dirichletBoundary(xi[idx],eta[idx])
                    Ddirichlet_X = dirichletBoundaryDerivativeX(xi[idx],eta[idx])
                    Ddirichlet_Y = dirichletBoundaryDerivativeY(xi[idx],eta[idx])
                    # Not anymore: In the line below the weigths have been taken out
                    F[(p+1)*(xbasisi-i+p) + ybasisi-j+q] += (fi*Ni_corr + (diCorrXi*(dx*dirichlet_xi_eta+d*Ddirichlet_X) + diCorrEta*(dy*dirichlet_xi_eta+d*Ddirichlet_Y)) - (Ddirichlet_X*diCorrXi + Ddirichlet_Y*diCorrEta))* gauss_weights[idx]
    return K,F

def elementChoose(model,p,q,knotvector_x, knotvector_y,i,j,etype=None):
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    """distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]"""
    points = torch.tensor(np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]]),dtype=torch.float32)
    distances = model(points)
    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point<0:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        Bspxi = Bspline(knotvector_x,p)
        Bspeta = Bspline(knotvector_y,q)
        if etype is not None: etype["inner"] +=1
        Ke, Fe = element(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta)
    elif outerElement:
        Ke = np.zeros(((p+1)*(q+1),(p+1)*(q+1)))
        Fe = np.zeros((p+1)*(q+1))
        if etype is not None: etype["outer"] +=1
    else:
        Bspxi = Bspline(knotvector_x,p)
        Bspeta = Bspline(knotvector_y,q)
        Ke, Fe = boundaryElement(model,p,q,knotvector_x, knotvector_y,i,j,Bspxi,Bspeta)
        if etype is not None: etype["boundary"] +=1
    if etype is not None: return Ke, Fe, etype
    return Ke, Fe
def elementTypeChoose(knotvector_x, knotvector_y,i,j,etype=None):
    x1 = knotvector_x[i]
    x2 = knotvector_x[i+1]
    y1 = knotvector_y[j]
    y2 = knotvector_y[j+1]
    """distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]"""
    points = torch.tensor(np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]]),dtype=torch.float32)
    distances = model(points)
    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point<0:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        if etype is not None: etype["inner"] +=1
    elif outerElement:
        if etype is not None: etype["outer"] +=1
    else:
        if etype is not None: etype["boundary"] +=1
    if etype is not None: return etype
    return None
def assembly(K,F,Ke,Fe,elemx,elemy,p,q, xDivision, yDivision):
    l = len(Fe)
    idxs = []
    for idxx in range(p+1):
        for idxy in range(q+1):
            idxs.append((elemx-p)*(xDivision+p+1)+(elemy-q) +idxx*(xDivision+p+1)+idxy)
    for idxx,i in enumerate(idxs):
        for idxy,j in enumerate(idxs):
            K[i,j] += Ke[idxx,idxy]
    for idx, i in enumerate(idxs):
        F[i] += Fe[idx]
    return K,F

def solveWeak(K,F):
    zero_rows = np.all(K == 0, axis=1)
    zero_cols = np.all(K == 0, axis=0)
    zero_f = zero_rows
    # Remove zero rows and columns
    K_reduced = K[~zero_rows][:, ~zero_cols]
    F_reduced = F[~zero_f]
    u = np.zeros(len(F))
    #u_reduced = np.dot(np.linalg.inv(K_reduced),F_reduced)
    #inv = np.linalg.inv(K_reduced)
    u_reduced = np.linalg.solve(K_reduced,F_reduced)
    #pinv = np.dot(np.linalg.inv(np.dot(np.transpose(K_reduced),K_reduced)),K_reduced)
    #svd_inv = svd_inverse(K_reduced)
    #reg_inv = regularized_inverse(K_reduced)
    #u_reduced = np.dot(inv,F_reduced)
    u[~zero_f] = u_reduced
    return u

def visualizeResultsBspline(model,results,p,q,knotvector_x, knotvector_y,surfacepoints=None,larger_domain=True):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    if larger_domain:
        x = np.linspace(-1,1,40)
        y = np.linspace(-1,1,40)
    else:
        x = np.linspace(0,1,40)
        y = np.linspace(0,1,40)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model)
            analitical.append(solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
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
    plt.show()
def calculateErrorBspline(model,results,p,q,knotvector_x, knotvector_y,larger_domain=True):
    xPoints = []
    yPoints = []
    result = []
    analitical = []
    if larger_domain:
        x = np.linspace(-1,1,40)
        y = np.linspace(-1,1,40)
    else:
        x = np.linspace(0,1,40)
        y = np.linspace(0,1,40)
    for xx in x:
        for yy in y:
            sum = 0
            xPoints.append(xx)
            yPoints.append(yy)
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            analitical.append(solution_function(xx,yy) if d>=0 else 0)
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            result.append(sum)
    MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    print(f"MSE: {MSE}")
    return MSE
def plotErrorHeatmap(model,results,knotvector_x,knotvector_y,p,q,larger_domain = True,N=150):

    marg = 0.1
    if larger_domain:
        x = np.linspace(-1-marg,1+marg,N)
        y = np.linspace(-1-marg,1+marg,N)
    else:
        x = np.linspace(0-marg,1+marg,N)
        y = np.linspace(0-marg,1+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_N=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model).detach().numpy()
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
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
    if larger_domain:
        x = np.linspace(-1-marg,1+marg,N)
        y = np.linspace(-1-marg,1+marg,N)
    else:
        x = np.linspace(0-marg,1+marg,N)
        y = np.linspace(0-marg,1+marg,N)
    X,Y = np.meshgrid(x,y)
    Z_A=np.zeros((N,N))
    Z_N=np.zeros((N,N))
    ERR=np.zeros((N,N))
    for idxx,xx in enumerate(x):
        for idxy, yy in enumerate(y):
            sum = 0
            d = mesh.distanceFromContur(xx,yy,model)
            Z_A[idxx,idxy] = solution_function(xx,yy) if d>=0 else 0
            
            for xbasis in range(len(knotvector_x)-p-1):
                for ybasis in range(len(knotvector_y)-q-1):
                    sum += B(xx,p,xbasis,knotvector_x)*B(yy,q,ybasis,knotvector_y)*results[(len(knotvector_x)-p-1)*xbasis+ybasis]
            sum = d*sum
            #sum = 0
            sum += (1-d)*dirichletBoundary(xx,yy)
            if d<0: sum = 0
            Z_N[idxx,idxy] = sum
    #MSE = (np.square(np.array(result)-np.array(analitical))).mean()
    #print(f"MSE: {MSE}")
    #return MSE
    ERR = np.abs(Z_N-Z_A)
    plt.contourf(X, Y, Z_A,levels=20)
    #plt.axis('equal')
    plt.colorbar()
    if larger_domain:
        highlight_level = 0.0
        plt.contour(X, Y, Z_A, levels=[highlight_level], colors='red') 
    else:
        points = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0),(1.0,0.5),(0.5,0.5),(0.5,1.0),(0.0, 1.0)]

        # Plot red lines between the points
        for i in range(len(points)-1):
                plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'r-')

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
                    etype = elementTypeChoose(knotvector_u,knotvector_w,elemx,elemy,etype)
            etypes.append(etype)
    print(etypes)
