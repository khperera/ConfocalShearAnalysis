"""
A class that holds information about images, including the image and its data. 
"""

import numpy as np
import numpy.typing as npt

class ImageHolder:
    """a class that holds an image and information about an image.
    dictionary must be of form.
    {"ImageType" : "Default", "Name" : "Default", "ZPos": -1, "Time": -1,"fit_data": {}}
    FIt data will be of form {Position : {}, Connections:{}}
    """
    def __init__(self, img: npt.ArrayLike = np.zeros((1), dtype=np.uint8),
                image_type: str = "Raw", name: str = "Default", z_position: int = -1,
                time: int = -1, fit_data: dict = None) -> None:
        if fit_data is None:
            self.img_info = {"image_type" : image_type, "name" : name,
                         "z_position" : z_position, "time": time,
                          "fit_data" : {"position":{},"connections":{}}}
        else:
            self.img_info = {"image_type" : image_type, "name" : name,
                         "z_position" : z_position, "time": time, "fit_data" : fit_data}
        self.img = img
        self.img_original = np.zeros((1,1,3), dtype=np.uint8)
        self.store_original_img()

    def __del__(self) -> None:
        """deletes the image and information about the image"""
        del self.img
        del self.img_info

    def store_image(self, img: npt.ArrayLike, image_type: str = None) -> None:
        """takes in a numpy array that represents an image, and what type it 
        is transformed to"""
        self.img = img
        if image_type is not None:
            self.img_info["image_type"] = image_type

    def store_image_info(self, image_type: str = None, name: str = None, z_position: int = None,
                time: int = None, fit_data: dict = None) -> None:
        """Stores information about image as a dict"""
        if image_type is not None:
            self.img_info["image_type"] = image_type
        if name is not None:
            self.img_info["name"] = name
        if z_position is not None:
            self.img_info["z_position"] = z_position
        if time is not None:
            self.img_info["time"] = time
        if fit_data is not None:
            self.img_info["fit_data"] = fit_data
    def return_fit_dictionary(self) -> dict:
        """Returns the dictionary to be edited"""
        return self.img_info["fit_data"]

    def return_image(self) ->  npt.ArrayLike:
        """Returns the image"""
        return self.img
    def return_original_image(self) ->  npt.ArrayLike:
        """Returns the original image"""
        return self.img_original

    def store_original_img(self) -> None:
        """Stores original image, just in case you want to utilize original image."""
        self.img_original = self.img

    def return_image_info(self) ->  dict:
        """returns the information about an image"""
        return self.img_info

    def return_image_size(self) ->  list:
        """returns the size of image"""
        return self.img.shape
