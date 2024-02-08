import json
import numpy as np
from mayavi import mlab
from src.utils import tools


def mayavi_sphere_points3d(particle_data):
    fig = mlab.figure('Mayavi 3D Spheres Points3D', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    x, y, z, r,i = np.array(particle_data).T
    r_scaled = r*2
    # Plot the sphere using Mayavi's points3d
    #mlab.sphere(x=x, y=y, z=z, radius=r, color=(np.random.random(), np.random.random(), np.random.random()), opacity=0.5)
    mlab.points3d(x, y, z, r_scaled,scale_factor = 1)

    # Add axes for reference
    mlab.axes()

    # Show the Mayavi plot
    mlab.show()

def load_from_json(json_file_location):
    config = tools.load_config(config_file_path=json_file_location)
    particle_data = config["3d_stack"]["fit_data"]["position"]
    calculate_particle_fract(particle_data=particle_data)
    mayavi_sphere_points3d(particle_data=particle_data)



def calculate_particle_fract(particle_data):
    x, y, z, r = np.array(particle_data).T
    area = 0
    
    average_r = 20

    x_min = x.min()+average_r*2
    x_max = x.max()-average_r*2
    y_min = y.min()+average_r*2
    y_max = y.max()-average_r*2
    z_min = z.min()+average_r*2
    z_max = z.max()-average_r*2

    between_bounds = (x<x_max)& (x>x_min)& (y>y_min)& (y<y_max)& (z>z_min)& (z<z_max)

    area = 3.14*r**3*4/3
    area_sum = np.sum(area[between_bounds])

    area_total = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)

    areapercent = area_sum/area_total

    print(areapercent)
