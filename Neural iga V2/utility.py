import math
def PlotSubdivide(x1,x2,y1,y2,ax,level,MAXLEVEL=2):
    halfx = (x1+x2)/2
    halfy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    gpx1 = halfx-w/2 *(1 / math.sqrt(3))
    gpy1 = halfy-h/2 *(1 / math.sqrt(3))

    gpx2 = halfx+w/2 *(1 / math.sqrt(3))
    gpy2 = halfy+h/2 *(1 / math.sqrt(3))
    
    if isBoundary(x1,x2,y1,y2) and level<MAXLEVEL:
        plotRectangle(x1, halfx,y1,halfy,level,ax)
        plotRectangle(x1, halfx,halfy,y2,level,ax)
        plotRectangle(halfx, x2,y1,halfy,level,ax)
        plotRectangle(halfx, x2,halfy,y2,level,ax)
    else:
        if not isOutside(x1,x2,y1,y2):
            ax.plot(gpx1,gpy1,'bx')
            ax.plot(gpx1,gpy2,'bx')
            ax.plot(gpx2,gpy1,'bx')
            ax.plot(gpx2,gpy2,'bx')
        return 0
    if level == MAXLEVEL:
        if not isOutside(x1,x2,y1,y2):
            ax.plot(gpx1,gpy1,'bx')
            ax.plot(gpx1,gpy2,'bx')
            ax.plot(gpx2,gpy1,'bx')
            ax.plot(gpx2,gpy2,'bx')
    else:
        PlotSubdivide(x1,halfx,y1,halfy,ax,level+1,MAXLEVEL)
        PlotSubdivide(x1,halfx,halfy,y2,ax,level+1,MAXLEVEL)
        PlotSubdivide(halfx,x2,y1,halfy,ax,level+1,MAXLEVEL)
        PlotSubdivide(halfx,x2,halfy,y2,ax,level+1,MAXLEVEL)
    

def plotRectangle(x1,x2,y1,y2,level,ax):
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
def isBoundary(x1,x2,y1,y2):
    distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]

    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point>1:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        return False
    elif outerElement:
        return False
    else:
        return True
def isOutside(x1,x2,y1,y2):
    distances = [x1**2 + y1**2,
                 x1**2 + y2**2,
                 x2**2 + y1**2,
                 x2**2 + y2**2]

    innerElement = True # all points are inside the body
    outerElement = True # all points are outside the body
    for point in distances:
        if point>1:
            innerElement = False
        else:
            outerElement = False
    if innerElement: #regular element
        return False
    elif outerElement:
        return True
    else:
        return False
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
circle = plt.Circle((0, 0), 1, facecolor='r')
ax.add_patch(circle)

# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Set limits for x and y axes
ax.set_xlim(0.5, 0.875)
ax.set_ylim(0.5, 0.875)
PlotSubdivide(0,1,0,1,ax,0,5)
plt.show()
