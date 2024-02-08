"""
Class that is used to segment 3D images. Must feed in unblurred images.
"""
import numpy as np
import numpy.typing as npt
from skimage import filters
from skimage import img_as_ubyte
from skimage.transform import rescale as scale1
import skimage as ski
import matplotlib.pyplot as plt
from src.utils.particlefinder_nogpu import MultiscaleBlobFinder, OctaveBlobFinder, get_deconv_kernel
from src.utils import rescale
from src.core import holder, exporter
from src.utils import tools


class ImageSegment():

    """
    #this will be a class that is init once and will take in files indivitually to do edits.
    #It interfaces with the image holder app by taking it in, 
    applying relevant segmenting information,
    #then overwriting with segmented image.
    #segmentation instruction passed via config file. 
    Should have option to call segmentation without config?
    """

    def __init__(self,config_file_path: str = "./config/testingconfig.json") -> None:
        #load in configuration files.
        config = tools.load_config(config_file_path=config_file_path)

        self.display_original_img = config["segment_config"]["display_original_img"]
        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)
        self.img_original = np.zeros((1,1,3), dtype=np.uint8)
        self.img_shape = None
        self.particle_centers = None
        self.kernel = None
        self.config_file_path= config_file_path
####################################################################################################
#main function. callable by other functions, classes


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def apply_segmentation(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """Given the config file, applies segmentation across all images in stack"""

        self.img = image_holder.return_image()
        self.squish_axis()
        self.find_particle_centers()
        dictionary_fit = image_holder.return_fit_dictionary()
        dictionary_fit["position"] = self.particle_centers

    def return_particledata(self):
        """returns the particle center data"""
        return self.particle_centers

####################################################################################################
#filters

    def create_kernel(self) -> None:
        """Creates deconvulusion kernel"""
        self.kernel = get_deconv_kernel(self.img, pxZ = 1, pxX=1)

    def find_particle_centers(self)-> None:
        """Uses prytrack to find particle centers"""
        self.convert_to_scikit_image()
        self.create_kernel()
        finder = MultiscaleBlobFinder(self.img.shape, Octave0=False, nbOctaves=4)
        centers = finder(self.img, removeOverlap=False,deconvKernel=self.kernel)
        print(centers)
        rescale_intensity = True
        if rescale_intensity:
            s = rescale.radius2sigma(centers[:,-2], dim=3)
            bonds, dists = rescale.get_bonds(positions=centers[:,:-2], radii=centers[:,-2], maxdist=3.0)
            brights1 = rescale.solve_intensities(s, bonds, dists, centers[:,-1])
            radii1 = rescale.global_rescale_intensity(s, bonds, dists, brights1)
        else:
            radii1 = centers[:,-2]
    
        particledata = list(zip(centers[:,0], centers[:,1], centers[:,2], radii1, centers[:,-1]))
        radius_threshold_high = 50
        radius_threshold_low = 0
        intensity_threshold_high = -.013

        particledata_filtered =  [item for item in particledata if (item[3] < radius_threshold_high and item[3] > radius_threshold_low and item[4]<intensity_threshold_high and 
                                                                    item[4]/item[3]>-.03)]
        self.particle_centers = particledata_filtered
        #plot particle data
        data_to_plot = [item[-1] for item in particledata_filtered]
        data_to_plot_2 = [item[-2] for item in particledata_filtered]
        data_to_plot_3 = [item[-1]/item[-2] for item in particledata_filtered]

        

        # Now plot the histogram
        plt.hist2d(data_to_plot, data_to_plot_2, bins = 100)  # Adjust bins as needed

        plt.title('Histogram of centers[:,-1]')
        

        plt.show()

        plt.hist(data_to_plot_3, bins = 100)
        plt.title('intensity/radii')
        

        plt.show()




        return particledata


    def save_cuts(self, x_divs = 50, y_divs = 50, z_divs = 50):
        """Call after segmenting and getting particle data
        Saves cuts and displays circles on them, saves to holder then exports"""
        img_size = self.img.shape
        print(img_size)
        x_size = img_size[2]/x_divs
        y_size = img_size[1]/y_divs
        z_size = img_size[0]/z_divs

        saver = exporter.ImageExporter(config_file_path=self.config_file_path)
        #need to find particle that are in plane with the cut and find radius that is in plane.
        i = 1
        #x_cuts
        for x_div in range(x_divs):
            pixel_dim = int(x_div*x_size)
            yz_cut = self.img[:,:,pixel_dim].copy()
            name = "yzcut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(x=pixel_dim)
            holder_yz = holder.ImageHolder(yz_cut,image_type="3D_cut",name= name)
            saver.save_3d_cuts(holder_yz,circles_in_dim)

        for y_div in range(y_divs):
            pixel_dim = int(y_div*y_size)
            xz_cut = self.img[:,pixel_dim].copy()
            name = "xzcut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(y=pixel_dim)
            holder_xz = holder.ImageHolder(xz_cut,image_type="3D_cut",name= name)
            saver.save_3d_cuts(holder_xz,circles_in_dim)

        for z_div in range(z_divs):
            pixel_dim = int(z_div*z_size)
            xy_cut = self.img[pixel_dim].copy()
            name = "xycut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(z=pixel_dim)
            holder_xy = holder.ImageHolder(xy_cut,image_type="3D_cut",name= name)
            saver.save_3d_cuts(holder_xy,circles_in_dim)

    def squish_axis(self, x_squish: float = 0.75, y_squish: float = 0.75, z_squish: float =1):
        self.img = scale1(self.img, (z_squish,x_squish,y_squish), anti_aliasing = True )







####################################################################################################
#utils


    def convert_to_single_channel(self, channel = 1):
        """
        #checks to see if an image is of the type uint8. Chooses the correct channel as well
        #channel 0,1,2 is b,g,r
        """
        self.img = np.uint8(cv2.split(self.img)[channel])

    def convert_to_scikit_image(self):
        """Converts an opencv image to scikit-image version, single channel"""
        ski.util.img_as_float(self.img)

    def convert_to_opencv_image(self):
        """Converts a scikit image to opencv image, single channel"""
        self.img = img_as_ubyte(self.img)

    def circles_in_dimension(self,x: int = None,y: int = None,z: int = None)-> npt.ArrayLike:
        """Returns the list of circles that are in the dimension and 
        gives effect radius at that plane"""
        positions = np.array(self.particle_centers)

        if x is not None:
            
            x_values = positions[:,0]
            r_values = positions[:, 3]
            filtered_positions = positions[np.abs(x_values-x) - r_values <= 0]
            filtered_positions[:,3] = np.sqrt(filtered_positions[:,3]**2 - (filtered_positions[:,0]-x)**2)
            
            return np.delete(filtered_positions,0, axis = 1)
        if y is not None:
            
            y_values = positions[:,1]
            r_values = positions[:, 3]
            filtered_positions = positions[np.abs(y_values-y) - r_values <= 0]
            filtered_positions[:,3] = np.sqrt(filtered_positions[:,3]**2 - (filtered_positions[:,1]-y)**2)
            return np.delete(filtered_positions,1, axis = 1)
        if z is not None:
            
            z_values = positions[:,2]
            r_values = positions[:, 3]
            filtered_positions = positions[np.abs(z_values-z) - r_values <= 0]
            filtered_positions[:,3] = np.sqrt(filtered_positions[:,3]**2 - (filtered_positions[:,2]-z)**2)
            return np.delete(filtered_positions,2, axis = 1)

        return positions