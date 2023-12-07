"""
A class that holds information about images, including the image and its data. 
"""

import numpy as np
import numpy.typing as npt


class ImageHolder:
    """a class that holds an image and information about an image.
    dictionary must be of form 
    {"ImageType" : "Default", "Name" : "Default", "ZPos": -1, "Time": -1,"Position Data": {}}
    """
    def __init__(self, img: npt.ArrayLike = np.zeros((1), dtype=np.uint8),
                    img_data: dict = None) -> None:
        if img_data is None:
            self.img_info = {"ImageType" : "Default", "Name" : "Default",
                             "ZPos": -1, "Time": -1,"Position Data":{}}
        else:
            self.img_info = img_data
        self.img = img

    def __del__(self) -> None:
        """deletes the image and information about the image"""
        del self.img
        del self.img_info

    def store_image(self, img: npt.ArrayLike, img_info: dict =  None) -> None:
        """takes in a numpy array that represents an image"""
        self.img = img
        if not img_info is None:
            self.img_info = img_info

    def store_image_info(self, img_info: dict =  None) -> None:
        """Stores information about image as a dict"""
        self.img_info = img_info

    def return_image(self) ->  npt.ArrayLike:
        """Returns the image"""
        return self.img

    def return_image_info(self) ->  dict:
        """returns the information about an image"""
        return self.img_info

    def return_image_size(self) ->  list:
        """returns the size of image"""
        return self.img.shape
