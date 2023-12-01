import cv2
import json
import os
import numpy as np

class imageSegment():
    def __init__(self,config_file_path = "./config/segmentingConfig.json"):
        
        #load in configuration files. 
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as file:
            config = json.load(file)

        self.imageSaveLocationBase = config["ImageSaver"]["DataStorageLocation"]

        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)



    


