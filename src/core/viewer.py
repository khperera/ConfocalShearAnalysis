import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




list_center = [(1,2,3),(-10,-10,6), (5,5,6)]
list_radius = [1,2,1]

def plt_sphere(particle_data,fig):
    ax = fig.add_subplot(projection='3d')
    for x1,y1, z1, r in particle_data:
        
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)

        ax.plot_surface(x-x1, y-y1, z-z1, color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)

def plotfigure(particle_data):
    fig = plt.figure()
    plt_sphere(particle_data,fig)  
    plt.show()