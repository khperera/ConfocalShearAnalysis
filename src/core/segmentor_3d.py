"""
Class that is used to segment 3D images. Must feed in unblurred images.
"""
import numpy as np
import numpy.typing as npt
from skimage import filters
from skimage import img_as_ubyte
import skimage as ski
from src.utils.particlefinder_nogpu import MultiscaleBlobFinder, OctaveBlobFinder, get_deconv_kernel
from src.utils import rescale
from src.core import holder
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
####################################################################################################
#main function. callable by other functions, classes


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def apply_segmentation(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """Given the config file, applies segmentation across all images in stack"""

        self.img = image_holder.return_image()
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
        self.kernel = get_deconv_kernel(self.img, pxZ = 9, pxX=0.579)

    def find_particle_centers(self)-> None:
        """Uses prytrack to find particle centers"""
        self.convert_to_scikit_image()
        self.create_kernel()
        finder = MultiscaleBlobFinder(self.img.shape, Octave0=False, nbOctaves=4)
        centers = finder(self.img, maxedge=-1,deconvKernel=self.kernel)
        rescale_intensity = True
        if rescale_intensity:
            s = rescale.radius2sigma(centers[:,-2], dim=2)
            bonds, dists = rescale.get_bonds(positions=centers[:,:-2], radii=centers[:,-2], maxdist=3.0)
            brights1 = rescale.solve_intensities(s, bonds, dists, centers[:,-1])
            radii1 = rescale.global_rescale_intensity(s, bonds, dists, brights1)

        particledata = list(zip(centers[:,0], centers[:,1], centers[:,2], radii1))
        self.particle_centers = particledata

        return particledata


    def save_cuts(self, X= 50, Y = 50, Z = 25):
        """Saves cuts to folder and displays circles on them"""
        z_cut = self.img











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
