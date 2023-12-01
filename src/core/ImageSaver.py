import json
import os
import cv2
#import data

#saves an imageHolder to a folder
class ImageSaver:
    def __init__(self,config_file_path = "./data/config.json"):
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as file:
            config = json.load(file)

        self.imageSaveLocationBase = config["ImageSaver"]["DataStorageLocation"]



    #saves an image to folder given an imageholder. Saves to a location dependent on imageholder type
    def saveImage(imageHolder):

        imgInfo = imageHolder.returnImageInfo()
        img = imageHolder.returnImage()

        imageType = str(imgInfo["ImageType"])
        name = str(imgInfo["Name"])
        saveLocation = self.imageSaveLocationBase+imageType+"Image/"+name+".tiff"


        cv2.imwrite(saveLocation,img)
