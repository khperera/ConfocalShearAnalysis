"""
Class that is used to segment images.
"""
import cv2
import numpy as np
import numpy.typing as npt
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

        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)

####################################################################################################
#main function. callable by other functions, classes


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def apply_segmentation(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """Given the config file, applies segmentation across all images in stack"""
        self.img = image_holder.return_image()
        img_info = image_holder.return_image_info()
        img_info["ImageType"] = "Segment"
        self.convert_to_single_channel(channel = 2)
        if self.bilateral_filter_marker:
            self.bilateral_filter(**self.bilateral_filter_parameters)
        if self.adaptive_filter_marker:
            self.adaptive_threshold(**self.adaptive_filter_parameters)
        image_holder.store_image(self.img,img_info)

####################################################################################################
#filters


    #bilateral filter wrapper
    def bilateral_filter(self, d: int = 10, sigma_color: int = 75,
                         sigma_space: int = 75) -> npt.ArrayLike:
        """
        Applies the bilateral filter function to an image.
        """
        self.img = cv2.bilateralFilter(self.img, d=d, sigmaColor=sigma_color,
                                       sigmaSpace=sigma_space)

    # adaptive threshold wrapper
    def adaptive_threshold(self, max_value: int = 255, block_size: int = 401, c: int = 10):
        """Applies openCV's adaptive threshold method"""
        self.img = cv2.adaptiveThreshold(self.img,
                                      max_value,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      block_size, c)

####################################################################################################
#utils


    def convert_to_single_channel(self, channel = 1):
        """
        #checks to see if an image is of the type uint8. Chooses the correct channel as well
        #channel 0,1,2 is b,g,r
        """
        self.img = np.uint8(cv2.split(self.img)[channel])
