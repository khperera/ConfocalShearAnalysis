"""
Module for developing a class to export files to user specifed location
"""

import json
import os
import cv2


class ImageExporter:
    """class for saving image from an image holder to a location."""
    def __init__(self,config_file_path = "./config/defaultconfig.json"):

        #this should be converted to a module
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.imageSaveLocationBase = config["DataSaveLocation"]

    #saves an image to folder given an imageholder. Saves to a location dependent on imageholder type. Returns if saved
    def saveImage(self, imageHolder):

        imgInfo = imageHolder.returnImageInfo()
        img = imageHolder.returnImage()

        imageType = str(imgInfo["ImageType"])
        name = str(imgInfo["Name"])

        saveDir = self.imageSaveLocationBase+str(imageType)+"Image/"#+name+".tiff"
        self.makeDir(saveDir)
        saveLocation = saveDir+name+".tiff"
        return cv2.imwrite(saveLocation,img)

    #Checks to see if a directory exists. If not, make a directory here. Returns true if exists
    def makeDir(self,directoryLocation):
        dirExist = os.path.exists(directoryLocation)

        if dirExist:
            return True
        else:
            os.makedirs(directoryLocation)
            return False
