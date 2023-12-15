import numpy as np
from mayavi import mlab


def mayavi_sphere_points3d(particle_data):
    fig = mlab.figure('Mayavi 3D Spheres Points3D', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    for x1, y1, z1, r in particle_data:
        # Create parametric values for a sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)

        # Plot the sphere using Mayavi's points3d
        mlab.points3d(x + x1, y + y1, z + z1, scale_factor=1, color=(np.random.random(), np.random.random(), np.random.random()), opacity=0.5)

    # Add axes for reference
    mlab.axes()

    # Show the Mayavi plot
    mlab.show()

