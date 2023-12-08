"""
Module for developing a class to export files to user specifed location
"""

import json
import os
import cv2
import numpy.typing as npt
from src.core import holder


class ImageExporter:
    """class for saving image from an image holder to a location."""
    def __init__(self,config_file_path = "./config/defaultconfig.json") -> None:

        #this should be converted to a module
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.image_save_location = config["DataSaveLocation"]

    def save_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """saves an image to folder given an imageholder. 
        Saves to a location dependent on imageholder type. Returns True if saved"""

        img_info = image_holder.returnImageInfo()
        img = image_holder.returnImage()

        image_type = str(img_info["ImageType"])
        name = str(img_info["Name"])

        #generate names
        save_dir = self.image_save_location+str(image_type)+"Image/"
        save_location = save_dir + name + ".tiff"

        self.make_dir(save_dir)
        return cv2.imwrite(save_location,img)

    def make_dir(self, directory_location: str) -> None:
        """Checks to see if a directory exists. If not, make a directory here. """

        dir_exists = os.path.exists(directory_location)

        if not dir_exists:
            os.makedirs(directory_location)
            return False
        return True
        