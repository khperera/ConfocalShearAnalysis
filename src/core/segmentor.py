"""
Class that is used to segment images.
"""
import cv2
import numpy as np
import numpy.typing as npt
from skimage import filters
from skimage import img_as_ubyte
import skimage as ski
from src.utils.particlefinder_nogpu import MultiscaleBlobFinder
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
        self.bilateral_filter_marker = config["segment_config"]["bilateral_filter_marker"]
        self.adaptive_filter_marker = config["segment_config"]["adaptive_filter_marker"]
        self.bilateral_filter_parameters = config["segment_config"]["bilateral_filter_parameters"]
        self.adaptive_filter_parameters = config["segment_config"]["adaptive_filter_parameters"]
        self.canny_filter_marker = config["segment_config"]["canny_filter_marker"]
        self.canny_filter_parameters = config["segment_config"]["canny_filter_parameters"]
        self.hough_filter_parameters = config["segment_config"]["hough_filter_parameters"]
        self.circle_filter_marker = config["segment_config"]["circle_filter_marker"]
        self.histogram_equilization_parameters = config["segment_config"]["histogram_equilization_parameters"]
        self.histogram_equilization_marker = config["segment_config"]["histogram_equilization_marker"]

        self.display_original_img = config["segment_config"]["display_original_img"]
        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)
        self.img_original = np.zeros((1,1,3), dtype=np.uint8)
        self.img_shape = None
        
####################################################################################################
#main function. callable by other functions, classes


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def apply_segmentation(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """Given the config file, applies segmentation across all images in stack"""
        self.img = image_holder.return_image()
        self.img_original = self.img.copy()
        img_info = image_holder.return_image_info()
        self.img_shape = image_holder.return_image_size()
        img_info["image_type"] = "Segment"
        self.convert_to_single_channel(channel = 2)

        #self.difference_guassians(15,9)
        if self.histogram_equilization_marker:
            self.histogram_equilization(**self.histogram_equilization_parameters)
        #self.cutoff_threshold()
        #self.difference_guassians(sigma_1=5,sigma_2=11, simga_image= 9)
        if self.bilateral_filter_marker:
            self.bilateral_filter(**self.bilateral_filter_parameters)
        if self.canny_filter_marker:
            self.canny_filter(**self.canny_filter_parameters)

        if self.adaptive_filter_marker:
            self.adaptive_threshold(**self.adaptive_filter_parameters)

        image_holder.store_image(self.img,image_type="Segment")

        if self.circle_filter_marker:
            if self.hough_filter_parameters["method"] == 2:
                self.find_particle_centers()
            else:
                circle_data = self.apply_hough_circle(**self.hough_filter_parameters)
                if circle_data is not None:
                    dictionary_fit = image_holder.return_fit_dictionary()
                    dictionary_fit["position"] = circle_data.tolist()
                    #self.img = image_holder.return_original_image()
                    #self.convert_to_single_channel(channel = 2)
                    self.draw_hough_circles(circle_data)

        if self.display_original_img:
            image_holder.store_image(self.img_original)
        else:
            image_holder.store_image(self.img)
       
####################################################################################################
#filters


    #bilateral filter wrapper
    def bilateral_filter(self, d: int = 10, sigma_color: int = 75,
                         sigma_space: int = 75) -> None:
        """
        Applies the bilateral filter function to an image.
        """
        self.img = cv2.bilateralFilter(self.img, d=d, sigmaColor=sigma_color,
                                       sigmaSpace=sigma_space)

    # adaptive threshold wrapper
    def adaptive_threshold(self, max_value: int = 255, block_size: int = 401, c: int = 10) -> None:
        """Applies openCV's adaptive threshold method"""
        self.img = cv2.adaptiveThreshold(self.img,
                                      max_value,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      block_size, c)

    def cutoff_threshold(self, threshold_value: int = 125) -> None:
        """Does a cutoff threshold below a certain value. Meaning, 0 below a certain brightness"""
        self.img = cv2.threshold(self.img, threshold_value, 255, cv2.THRESH_TOZERO)[1]

    def difference_guassians(self, sigma_1: int = 1, sigma_2: int = 3, simga_image: int = 3):
        """Applies difference of Guassians filter"""
        guassian1 = cv2.GaussianBlur(self.img,(sigma_1,sigma_1),0)
        guassian2 = cv2.GaussianBlur(self.img, (sigma_2,sigma_2),0)
        self.img = cv2.GaussianBlur(guassian2-guassian1, (simga_image,simga_image),0)

    def canny_filter(self, threshold_1: int = 10, threshold_2: int = 10) -> None:
        """Applies openCV's canny filter method"""
        self.img = cv2.Canny(self.img,threshold1=threshold_1,threshold2=threshold_2)

    def histogram_equilization(self, method: int = 0, cliplimit: float = 2.0,
                                tile_grid_size: int = 8) -> None:
        """Applies either histogram equilization or adaptive based on input parameters"""
        if method ==0 :
            self.img = cv2.equalizeHist(self.img)
        else:
            clahe = cv2.createCLAHE(clipLimit=cliplimit,
                                     tileGridSize=(tile_grid_size,tile_grid_size))
            self.img = clahe.apply(self.img)


    def apply_hough_circle(self, method: int = 1, dp: float = 1.5,
                     min_distance_ratio: int = 10, param_1: int = 200, param_2: int = 0.9,
                     min_radius_fraction = 0.01, max_radius_fraction: int = 0.05)-> npt.ArrayLike:
        """Extracts hough circles from image. Min radius is radius as fraction of image dimension
            Max radius is radius as fraction of image dimension. Data on fit stored in image props
            Takes in a smooth image, 7x7 kernal 
        """
        if method == 0:
            method_in = cv2.HOUGH_GRADIENT
        else:
            method_in = cv2.HOUGH_GRADIENT_ALT


        image_dimension = self.img_shape[0]

        min_radius = int(image_dimension*min_radius_fraction)
        max_radius = int(image_dimension*max_radius_fraction)
        min_distance = int(min_radius*min_distance_ratio)

        circle_data = cv2.HoughCircles(image = self.img, dp = dp, method = method_in,
                                        minDist = min_distance, param1= param_1,
                                        param2= param_2, minRadius = min_radius,
                                        maxRadius = max_radius)
        if circle_data is None:
            return None
        else:
            return np.uint16(np.around(circle_data))


    def draw_hough_circles(self, list_circles: npt.ArrayLike = None) -> None:
        """Draws a Hough Circle on an image"""
        self.img = cv2.merge([self.img,self.img,self.img])
        for circle in list_circles[0,:]:
            center = (circle[0],circle[1])
            radius = circle[2]
            color = (0,255,0)
            thickness = 1
            cv2.circle(self.img, center, radius, color, thickness)

    def draw_pyrtrack_circles(self, centers: npt.ArrayLike = None, radii: npt.ArrayLike = None) -> None:
        """Draws circles made in pyrtrack"""
        if self.display_original_img:
            utilized_img = self.img_original
        else:
            utilized_img = cv2.merge([self.img,self.img,self.img])
        
        if radii is not None:
            particledata = zip(centers[:,0], centers[:,1], radii)
        else:
            particledata = zip(centers[:,0], centers[:,1], centers[:,2])


        for x,y,r in particledata:
            center = (int(x),int(y))
            radius = int(r)
            color = (0,255,0)
            thickness = 1
            cv2.circle(utilized_img, center, radius, color, thickness)
        if self.display_original_img:
            self.img_original = utilized_img
        else:
            self.img = utilized_img



    def find_particle_centers(self)-> None:
        """Uses prytrack to find particle centers"""
        self.convert_to_scikit_image()
        finder = MultiscaleBlobFinder(self.img.shape, Octave0=False, nbOctaves=4)
        centers = finder(self.img, maxedge=-1)
        rescale_intensity = True
        if rescale_intensity:
            s = rescale.radius2sigma(centers[:,-2], dim=2)
            bonds, dists = rescale.get_bonds(positions=centers[:,:-2], radii=centers[:,-2], maxdist=3.0)
            brights1 = rescale.solve_intensities(s, bonds, dists, centers[:,-1])
            radii1 = rescale.global_rescale_intensity(s, bonds, dists, brights1)

        self.convert_to_opencv_image()

        self.draw_pyrtrack_circles(centers,radii1)


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
