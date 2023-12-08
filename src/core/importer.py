"""
Imports images 
"""
import os
import cv2
import numpy as np
import numpy.typing as npt

class ImageImporter:
    """Class that imports images"""
    def __init__(self, image_location: str = "") -> None:
        self.image_location = image_location

    def read_location(self, location)-> None:
        """user command to give a location, then give an image bac"""
        self.image_location = location



    #the image reading function
    def return_image(self) -> npt.ArrayLike:
        """command to read file, if no file found, returns an empty image"""
        if not os.path.isfile(self.image_location):
            return np.zeros((1,1,3), dtype=np.uint8)
        else:
            return cv2.imread(self.image_location,1)
