import numpy as np



#a class that holds an image and information about an image.
class ImageHolder:

    def __init__(self, img = np.zeros((1), dtype=np.uint8), imgData = {"ImageType" : "Default", "Name" : "Default", "ZPos": -1, "Time": -1,"Position Data":{}}):
        self.img = img
        self.imgInfo = imgData

    #deletes the image and information about the image
    def __del__(self):
        del self.img
        del self.imgInfo

    #takes in a numpy array that represents an image
    def storeImage(self, img, imgInfo =  {"ImageType" : "Default", "Name" : "Default", "ZPos": -1, "Time": -1,"Position Data": {}}):
        self.img = img
        self.imgInfo = imgInfo

    #stores the image info
    def storeImageInfo(self,imgInfo =  {"ImageType" : "Default", "Name" : "Default", "ZPos": -1, "Time": -1,"Position Data":{}}):
        self.imgInfo = imgInfo

    #returns the image
    def returnImage(self):
        return self.img

    #returns the information about an image
    def returnImageInfo(self):
        return self.imgInfo
    
    #returns the size of each dimension of an image
    def returnImageSize(self):
        return self.img.shape
