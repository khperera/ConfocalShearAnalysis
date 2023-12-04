import cv2
import json
import os
import numpy as np
import ImageHolder

#this will be a class that is init once and will take in files indivitually to do edits.
#It interfaces with the image holder app by taking it in, applying relevant segmenting information,
#then overwriting with segmented image.

#segmentation instruction passed via config file. Should have option to call segmentation without config?
class imageSegment():
    def __init__(self,config_file_path = "./config/segementingConfig.json"):
        
        #load in configuration files. 
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as file:
            config = json.load(file)

        self.imageSaveLocationBase = config["DataStorageLocation"]

        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)

    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def applySegmentation(self, imageHolder = ImageHolder.ImageHolder()):
        self.img = imageHolder.returnImage()

        pass


    


