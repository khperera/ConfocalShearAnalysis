class imageGrabber:
    def __init__(self):


        self.imageLocation = ""


    #user command to give a location, then give an image back
    def returnImage(self,location):
        self.imageLocation = location
        return self.openImage()

    #the image reading function
    def openImage():
        return cv2.imread(self.imageLocation,1)
