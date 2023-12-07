import glob
from src.core import holder, segmentor, exporter, importer
import os
import json

class ImageCollection():
    """Class that holds images and applies operations to whole collection.
    #   Will be more specialized later (time data vs zstack)
    #class duties:
    #   Must have images as ImageHolders
    #   Must be able to import images given a location folder of images.
    #   Can be given ImageHolders, will be added to list.
    #   Imageholders are stored with relevant metadata data, sorted based on it. (Z position 
    #       for z stack, time for time series, number for unassociated data)
    #       need sorting operation
    #   Can save the imageholders given a folder location.
    #
    #things to consider for later:
    #   if a large amount of images, operations may need to be buffered, all images can't be stored at once. Can store via pickle
    #       Will be a function of how many images and size of images. Basically, will group images in groups of X. Operations will
    #       done sequentially on each set of objects, then stored, next set open."""

    def __init__(self, image_location: str  = "", save_location: str = "", list_image_locations: list = [], config_file_path: str = "./config/defaultconfig.json"):
        
        config_file_path = os.path.abspath(config_file_path)

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as file:
            config = json.load(file)


        #basic parameters, unpopulated or to be read in.
        if saveLocation == "":
            self.save_location = config["DataSaveLocation"]
        else:
            self.save_location = save_location

        if imageLocation== "":
            self.image_location = config["DataReadLocation"] 
        else:
            self.image_location = image_location
       
        self.list_of_image_locations = list_image_locations

        #holds the image holders
        #  relevant parameter : imageHolder Object
        self.image_storage = {}

        #

##########################################################
#Callable functions

    #loads images. Will load from folder, unless a list of specific images are given.
    def loadImages(self):
        if not self.list_of_image_locations:
            self.findFiles()
        
        for image in self.list_of_image_locations:
            self.insertImageIntoCollection(image)

    #applies a segmentation operation to all images in stack. Can give a custom config 
    def applySegmentation(self, config_file_path="./config/segementingConfig.json"):
        image_segmentor = segmentor.ImageSegment(config_file_path=config_file_path)

        for image in self.imageStorage:
            image_segmentor.applySegmentation(self.imageStorage[image])
        return True


    #adds an external image holder. Should take in relevant metadata in inhertied class 
    #   and add to dictionary with that
    def addImageHolder(self,imageHolder = ImageHolder.ImageHolder()):
        if not self.imageStorage:
            self.imageStorage[0]= imageHolder
        else:
            self.imageStorage[max(self.imageStorage)+1] = imageHolder

    #saves images to specified folder. Returns true if all files saved
    def saveFiles(self, savelocation = ""):
        if not savelocation == "":
            self.saveLocation = savelocation
        
        imageSaver = ImageExporter.ImageExporter(config_file_path="./config/testingconfig.json")
        truthStatement = True
        for image in self.imageStorage:
            truthStatement = truthStatement and imageSaver.saveImage(self.imageStorage[image])
        return truthStatement
    


############################################################
#Class utilies/helper functions

    #find files.
    def findFiles(self):
        self.listOfImageLocations = glob.glob(self.imageLocation+"*")

    #inserts image into the collection given a location. Puts relevant metadata in dictionary
    #   need to make a metadata generator.
    def insertImageIntoCollection(self,location):
        imageImporter = ImageImporter.ImageImporter(imageLocation=location)
        
        if not self.imageStorage:
            metaData = {"ImageType" : "Raw", "Name" : "0", "ZPos": -1, "Time": -1,}
            self.imageStorage[0] = ImageHolder.ImageHolder(imageImporter.returnImage(),metaData)
        else:
            num1 = max(self.imageStorage)+1
            metaData = {"ImageType" : "Raw", "Name" : str(num1), "ZPos": -1, "Time": -1,}
            self.imageStorage[num1] = ImageHolder.ImageHolder(imageImporter.returnImage(),metaData)
    
    #checks to see how many images are in the list.
    def checkImageLength(self):
        return len(self.imageStorage)
    
    