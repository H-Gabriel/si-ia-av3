import numpy as np
import matplotlib.pyplot as plt
from algoritmos import Hillclimb, LRS, GRS, SA

def f1(x,y):
    return np.square(x) + np.square(y)

def f2(x,y): #1.69, 1.69
    return np.exp(-(x**2 + y**2)) + 2*np.exp(-((x - 1.7)**2 + (y-1.7)**2))

def f3(x,y):
    return -20*np.exp(-0.2*np.sqrt((x**2 + y**2)/2)) - np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2) + 20 + np.exp(1)

def f4(x,y):
    return (x**2 - 10*np.cos(2*np.pi*x) + 10) + (y**2 - 10*np.cos(2*np.pi*y) + 10)

def f5(x,y):
    return (x-1)**2 + 100*(y - x**2)**2 # Verificar se esse plot está correto

def f6(x,y):
    return x*np.sin(4*np.pi*x) - y*np.sin(4*np.pi*y + np.pi) + 1

def f7(x,y):
    return -np.sin(x)*np.sin(x**2/np.pi)**20 - np.sin(y)*np.sin(2*y**2/np.pi)**20

def f8(x,y):
    return -(y+47)*np.sin(np.sqrt(np.absolute(x/2 + y + 47))) - x*np.sin(np.sqrt(np.absolute(x - y - 47)))

'''
x = np.linspace(0, np.pi, 500)
y = np.linspace(0, np.pi, 500)
xx, yy = np.meshgrid(x, y)
y = f7(xx, yy)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx,yy,y,rstride=10,cstride=10,alpha=0.6,cmap='jet')
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_zlim(-1.75, 0)
ax.scatter(2.2, 1.5, f7(2.2, 1.5), s=40, color='k')
plt.show()
'''

'''
lrs = LRS(f=f7, minimize=True, x_range=[0, np.pi], y_range=[0, np.pi])
for _ in range(10):
    lrs.run()
tempera = SA(f=f7, minimize = True, x_range=[-2,2], y_range=[-1,3])
for _ in range(10):
    tempera.run()

hillclimb = Hillclimb(f=f5, minimize=True, x_range=[-2,2], y_range=[-1,3], size = 4)
for _ in range(10):
    hillclimb.run()
'''

