import numpy as np
#a class that holds an image and information about an image.
class ImageHolder:

    def __init__(self):
        self.img = np.zeros((1), dtype=np.uint8)
        self.imgInfo = {"ImageType" : "None", "Name" : "None"}

    #deletes the image and information about the image
    def __del__(self):
        del self.img
        del self.imgInfo

    #takes in a numpy array that represents an image
    def storeImage(self, img):
        self.img = img

    #stores the image info
    def storeImageInfo(self,imgInfo):
        self.imgInfo = imgInfo

    #returns the image
    def returnImage(self):
        return self.img

    #returns the information about an image
    def returnImageInfo(self):
        return self.imgInfo
    
    def returnImageSize(self):
        return self.img.shape
