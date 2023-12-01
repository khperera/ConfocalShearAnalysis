import cv2
import numpy as np
import os
class ImageGrabber:
    def __init__(self):
        self.imageLocation = ""


    #user command to give a location, then give an image back
    def readLocation(self, location):
        self.imageLocation = location
        return 


    #the image reading function
    def returnImage(self):

        
        if not os.path.isfile(self.imageLocation):
            return np.zeros((1,1,3), dtype=np.uint8)
        else:
            return cv2.imread(self.imageLocation,1)
