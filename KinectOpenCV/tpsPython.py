import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as p
'''
def plotTPS(XYZ, gridPoints, subplot):
    GRID_POINTS = gridPoints
    x_min = XYZ[:,0].min()
    x_max = XYZ[:,0].max()
    y_min = XYZ[:,1].min()
    y_max = XYZ[:,1].max()
    xi = np.linspace(x_min, x_max, GRID_POINTS)
    yi = np.linspace(y_min, y_max, GRID_POINTS)
    XI, YI = np.meshgrid(xi, yi)
    rbf = Rbf(XYZ[:,0],XYZ[:,1],XYZ[:,2],function='thin-plate',smooth=0.0)
    ZI = rbf(XI,YI)

    subplot.plot_wireframe(XI, YI, ZI)
    subplot.scatter(XYZ[:,0],XYZ[:,1],XYZ[:,2], 'z', 40, 'r', True)
    '''
    
def tpsWarp3d(XYZ_old, XYZ_new):

    XY_dx = np.concatenate((XYZ_old[:,[0]], XYZ_old[:,[1]], (XYZ_new[:,[0]] - XYZ_old[:,[0]])), axis=1)
    print XY_dx    
    XY_dy = np.concatenate((XYZ_old[:,[0]], XYZ_old[:,[1]], (XYZ_new[:,[1]] - XYZ_old[:,[1]])), axis=1)
    XY_dz = np.concatenate((XYZ_old[:,[0]], XYZ_old[:,[1]], (XYZ_new[:,[2]] - XYZ_old[:,[2]])), axis=1)
    rbf_x = Rbf(XY_dx[:,0],XY_dx[:,1],XY_dx[:,2],function='thin-plate',smooth=0.0)
    rbf_y = Rbf(XY_dy[:,0],XY_dy[:,1],XY_dy[:,2],function='thin-plate',smooth=0.0)
    rbf_z = Rbf(XY_dz[:,0],XY_dz[:,1],XY_dz[:,2],function='thin-plate',smooth=0.0)
    return rbf_x, rbf_y, rbf_z

def warpPoints(XYZ_old, XYZ_new, X, Y, Z):
    rbf_x, rbf_y, rbf_z = tpsWarp3d(XYZ_old, XYZ_new)
    xi = rbf_x(X, Y) + X
    yi = rbf_y(X, Y) + Y
    zi = rbf_z(X, Y) + Z
    XI = np.array([xi]).T
    YI = np.array([yi]).T
    ZI = np.array([zi]).T
    #return np.concatenate((XI, YI, ZI), axis=1)
    return xi, yi, zi
XYZ = np.array([[100, 100,  0],
			[150, 100,  0],
			[200, 100,  0],
			[100, 150,  0],
			[150, 150, 0],
			[200, 150, 0],
			[100, 200,  0],
			[150, 200, 0],
			[200, 200, 0]
			])
XYZ2 = np.array([[120, 100,  0],
			[150, 130,  10],
			[180, 100,  0],
			[115, 150,  0],
			[150, 175, 10],
			[185, 150, 0],
			[115, 210,  0],
			[150, 215, 10],
			[183, 210, 0]
			])
   
#XYZ3 = np.concatenate((XYZ[:,[0]], XYZ[:,[1]], (XYZ2[:,[2]] - XYZ[:,[2]])), axis=1)
#XYZ3 = XYZ[:,0] + XYZ[:,1] + (XYZ2[:,2] - XYZ[:,2])
fig = plt.figure()
'''
s1 = fig.add_subplot(3,1,1, projection='3d', adjustable='box', aspect=1)
s2 = fig.add_subplot(3,1,2, projection='3d', adjustable='box', aspect=1)
s3 = fig.add_subplot(3,1,3, projection='3d', adjustable='box', aspect=1)
'''
subplot = fig.add_subplot(1,1,1, projection='3d', adjustable='box', aspect=1)
#plotTPS(XYZ, 15, subplot)
GRID_POINTS = 8
x_min = XYZ[:,0].min()
x_max = XYZ[:,0].max()
y_min = XYZ[:,1].min()
y_max = XYZ[:,1].max()
z_min = XYZ[:,2].min()
z_max = XYZ[:,2].max()
xi = np.linspace(x_min, x_max, GRID_POINTS)
yi = np.linspace(y_min, y_max, GRID_POINTS)
zi = np.linspace(z_min, z_max, GRID_POINTS)
g, ZI= np.meshgrid(xi, zi)
XI, YI, = np.meshgrid(xi, yi)
print ZI
#XI = np.array([xi]).T
#YI = np.array([yi]).T
#ZI = np.array([zi]).T
#XYZ_mesh = np.concatenate((XI, YI, ZI), axis=1)
#print XYZ_mesh
#mesh = warpPoints(XYZ, XYZ2, XI,YI,ZI)
#print mesh
X,Y,Z = warpPoints(XYZ, XYZ2, XI,YI,ZI)

#print X.shape
#print Y.shape
print Z.shape
print "before"
subplot.plot_wireframe(X, Y, Z)
print "done"
'''
rbf,a,b = tpsWarp3d(XYZ, XYZ2)
dx = rbf(XI,YI)
dy = a(XI,YI)
dz = b(XI,YI)
#subplot.plot_wireframe(XI, YI, dx)
#subplot.plot_wireframe(XI, YI, dy)
subplot.plot_wireframe(XI, YI, dz)
'''
subplot.scatter(XYZ[:,0],XYZ[:,1],XYZ[:,2], 'z', 40, 'r', True)
subplot.scatter(XYZ2[:,0],XYZ2[:,1],XYZ2[:,2], 'z', 40, 'g', True)
#plotTPS(XYZ, 10, s1)
#plotTPS(XYZ2, 10, s2)
#plotTPS(XYZ3, 10, s3)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



















'''
XYZ = np.array([[0, 0,  0],
			[1, 0,  0],
			[2, 0,  0],
			[0, 1,  0],
			[1, 1, 0],
			[2, 1, 0],
			[0, 2,  0],
			[1, 2, 0],
			[2, 2, 0]
			])
XYZ2 = np.array([[0, 0,  0],
			[1, 0,  1],
			[2, 0,  1],
			[0, 1,  0],
			[1, 1, -1],
			[2, 1, -1],
			[0, 2,  0],
			[1, 2, -1],
			[2, 2, -2]
			])
print (XYZ2[:,2] - XYZ[:,2])
XYZ3 = np.concatenate((XYZ[:,[0]], XYZ[:,[1]], (XYZ2[:,[2]] - XYZ[:,[2]])), axis=1)
#XYZ3 = XYZ[:,0] + XYZ[:,1] + (XYZ2[:,2] - XYZ[:,2])
print XYZ3
fig = plt.figure()

s1 = fig.add_subplot(3,1,1, projection='3d', adjustable='box', aspect=1)
s2 = fig.add_subplot(3,1,2, projection='3d', adjustable='box', aspect=1)
s3 = fig.add_subplot(3,1,3, projection='3d', adjustable='box', aspect=1)
plotTPS(XYZ, 10, s1)
plotTPS(XYZ2, 10, s2)
plotTPS(XYZ3, 10, s3)
plt.show()
'''
