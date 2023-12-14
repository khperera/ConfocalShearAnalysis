"""
Module for developing a class to export files to user specifed location
"""
import os
import json
import cv2
import numpy.typing as npt
from skimage import io,color
from src.core import holder
from src.utils import tools


class ImageExporter:
    """class for saving image from an image holder to a location."""
    def __init__(self,config_file_path = "./config/defaultconfig.json") -> None:

        config = tools.load_config(config_file_path)
        self.image_save_location = config["DataSaveLocation"]
        self.last_saved_location = ""

    def save_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """saves an image to folder given an imageholder. 
        Saves to a location dependent on imageholder type. Returns True if saved"""

        img_info = image_holder.return_image_info()
        img = image_holder.return_image()

        image_type = str(img_info["image_type"])
        name = str(img_info["name"])
        #generate names
        save_dir = self.image_save_location+str(image_type)+"Image/"
        self.last_saved_location = save_dir

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

    def save_json(self, property_dictionary: dict = None) -> None:
        """given a dictionary, saves the properties of an image collection to a file."""
        with open(self.last_saved_location+"collection_properties.json","w",encoding="utf-8") as f:
            json.dump(property_dictionary,f,ensure_ascii=False, indent = 4)
        

    def save_3d_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """Saves a 3D image as a tiff"""
        img_info = image_holder.return_image_info()
        img = image_holder.return_image()

        image_type = str(img_info["image_type"])
        name = "3Dstack"+str(img_info["name"])
        #generate names
        save_dir = self.image_save_location+str(image_type)+"Image/"
        self.last_saved_location = save_dir

        save_location = save_dir + name + ".tiff"

        self.make_dir(save_dir)
        io.imsave(save_location,img)