"""
Module for developing a class to export files to user specifed location
"""
import os
import cv2
import numpy.typing as npt
from src.core import holder
from src.utils import tools


class ImageExporter:
    """class for saving image from an image holder to a location."""
    def __init__(self,config_file_path = "./config/defaultconfig.json") -> None:

        config = tools.load_config(config_file_path)
        self.image_save_location = config["DataSaveLocation"]

    def save_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """saves an image to folder given an imageholder. 
        Saves to a location dependent on imageholder type. Returns True if saved"""

        img_info = image_holder.return_image_info()
        img = image_holder.return_image()

        image_type = str(img_info["image_type"])
       
        name = str(img_info["name"])
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
        