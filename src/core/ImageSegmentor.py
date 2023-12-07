import cv2
import json
import os
import numpy as np
from src.core import ImageHolder

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

        #self.imageSaveLocationBase = config["DataStorageLocation"]


        #self.smoothingParameters = {d}

        #temporary checks for filter. set to true for now. Will be set on or off via config
        #file
        self.bilateralFilterMarker = True
        self.adaptiveFilterMarker = True



        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)

####################################################################################################
#main function. callable by other functions, classes
    


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def applySegmentation(self, imageHolder = ImageHolder.ImageHolder()):

        self.img = imageHolder.returnImage()
        imgInfo = imageHolder.returnImageInfo()
        imgInfo["ImageType"] = "Segment"
        self.convertToSingleChannel(channel = 0)
        
        if self.bilateralFilterMarker:
            self.bilateralFilter()
        if self.adaptiveFilterMarker:
            self.adaptiveThreshold()
        

        imageHolder.storeImage(self.img,imgInfo)

####################################################################################################
#filters


    #bilateral filter wrapper
    def bilateralFilter(self, params = {"d":10,"sigmaColor" :  75, "sigmaSpace" : 75}):
        d,sigmaColor,sigmaSpace = params["d"],params["sigmaColor"],params["sigmaSpace"]
        self.img = cv2.bilateralFilter(self.img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    # adaptive threshold wrapper
    def adaptiveThreshold(self,params = {"maxValue":255,"blockSize":401,"C":10}):

        maxValue = params["maxValue"]
        blockSize = params["blockSize"]
        C = params["C"]

        self.img = cv2.adaptiveThreshold(self.img,
                                      maxValue,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      blockSize, C)
        

# adaptive threshold wrapper
    def adaptiveThreshold(self,params = {"maxValue":255,"blockSize":401,"C":10}):

        maxValue = params["maxValue"]
        blockSize = params["blockSize"]
        C = params["C"]

        self.img = cv2.adaptiveThreshold(self.img,
                                      maxValue,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      blockSize, C)
####################################################################################################
#utils

#chekcs to see if an image is of the type uint8. Chooses the correct channel as well
    #channel 0,1,2 is b,g,r
    def convertToSingleChannel(self, channel = 0):
        self.img = np.uint8(cv2.split(self.img)[channel])
