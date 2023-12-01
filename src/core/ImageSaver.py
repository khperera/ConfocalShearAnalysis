import json
import os
import cv2
#import data

#saves an imageHolder to a folder
class ImageSaver:
    def __init__(self,config_file_path = "./config/config.json"):
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as file:
            config = json.load(file)

        self.imageSaveLocationBase = config["ImageSaver"]["DataStorageLocation"]


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
