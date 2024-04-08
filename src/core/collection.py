"""
Class that oversees image operations
#things to consider for later:
if a large amount of images, operations may need to be buffered, 
all images can't be stored at once. Can store via pickle
Will be a function of how many images and size of images. 
Basically, will group images in groups of X. Operations will
done sequentially on each set of objects, then stored, next set open.
"""

import glob
import skimage as ski
import numpy as np
import numpy.typing as npt
from src.core import holder, segmentor, exporter, importer, segmentor_3d
from src.utils import tools


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
    """

    def __init__(self, image_location: str  = "", save_location: str = "",
                list_image_locations: list = None,
                config_file_path: str = "./config/defaultconfig.json"):

        config = tools.load_config(config_file_path=config_file_path)
        #basic parameters, unpopulated or to be read in.
        self.config_file_path = config_file_path
        if save_location == "":
            self.save_location = config["DataSaveLocation"]
        else:
            self.save_location = save_location

        if image_location== "":
            self.image_location = config["DataReadLocation"]
        else:
            self.image_location = image_location

        if list_image_locations is None:
            self.list_of_image_locations = []
        else:
            self.list_of_image_locations = list_image_locations

        #holds the image holders
        #  relevant parameter : imageHolder Object
        self.image_storage = {}
        self.stack_3d = None
        self.holder_3d = None
        self.particle_data3d = None
        #self.total_files = 0

##########################################################
#Callable functions

    def load_images(self) -> None:
        """loads images. Will load from folder, unless a list of specific images are given."""
        if not self.list_of_image_locations:
            self.find_files()

        for image in self.list_of_image_locations:
            self.insert_image_to_collection(image)


    def apply_segmentation(self, segment_config_file_path = None):
        """applies a segmentation operation to all images in stack. Can give a custom config"""
        if segment_config_file_path is None:
            config_file_path_segment = self.config_file_path
        image_segmentor = segmentor.ImageSegment(config_file_path=config_file_path_segment)

        for image in self.image_storage.values():
            image_segmentor.apply_segmentation(image)
        return True


    def add_image_holder(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """adds an external image holder. Should take in relevant metadata in inhertied class """
        if not self.image_storage:
            self.image_storage[0]= image_holder
        else:
            self.image_storage[max(self.image_storage)+1] = image_holder

    def save_files(self, save_location = ""):
        """saves images to specified folder. Returns true if all files saved. Saves the 
        config file as well
        """
        if save_location != "":
            self.save_location = save_location
        

        config_file_path_save = self.config_file_path
        image_saver = exporter.ImageExporter(config_file_path=config_file_path_save)

        #dictionary of image properties.
        image_dictionary = {}
        

        truth_statement = True

        for image in self.image_storage.values():
            image_properties = image.return_image_info()
            image_name = image_properties["name"]
            truth_statement = truth_statement and image_saver.save_image(image)
            image_dictionary[image_name] = image_properties
        
        if self.holder_3d is not None:
            image_dictionary["3d_stack"] = self.holder_3d.return_image_info()
            image_saver.save_3d_image(self.holder_3d)


        image_saver.save_json(property_dictionary=image_dictionary)

        return truth_statement
    
    def extract_numpy_imgs(self) -> npt.ArrayLike:
        """Extracts the images from image holders"""
        img_stack = []
        for image in self.image_storage.values():
            img_stack.append(ski.util.img_as_float(image.return_image()))

        return img_stack
    
    def convert_to_3d(self) -> holder.ImageHolder:
        """Converts the image stack to a scikit 3D image array"""
        self.stack_3d = np.dstack(tuple(self.extract_numpy_imgs())).transpose(2,0,1)
        print(self.stack_3d.shape)
        self.holder_3d = holder.ImageHolder(img = self.stack_3d,image_type="Segment")

      
    def segment_3d_image(self, segment_config_file_path = None):
        """applies a segmentation operation to 3D stack. Can give a custom config"""
        if segment_config_file_path is None:
            config_file_path_segment = self.config_file_path
        image_segmentor = segmentor_3d.ImageSegment(config_file_path=config_file_path_segment)
        image_segmentor.apply_segmentation(self.holder_3d)
        self.particle_data3d = image_segmentor.return_particledata()
        image_segmentor.save_cuts()
        #image_segmentor.napari_viewer()

    def return_3d_particledata(self):
        """returns the 3d particle data"""
        return self.particle_data3d

############################################################
#Class utilies/helper functions

    #find files.
    def find_files(self):
        """Finds files in defined folder location from config file"""
        self.list_of_image_locations = glob.glob(self.image_location+"*")

    def insert_image_to_collection(self,location):
        """inserts image into the collection given a location. 
            Puts relevant metadata in dictionary"""
        image_importer = importer.ImageImporter(image_location=location)

        #adds images to collection based on order added.
        #Later, will be based on some parameter, like Zpos
        if not self.image_storage:
            metadata = {"img" : image_importer.return_image(),
                        "image_type" : "Raw", "name" : "0", "z_position": -1, "time": -1,}
            self.image_storage[0] = holder.ImageHolder(**metadata)
        else:
            num1 = max(self.image_storage)+1
            metadata = {"img": image_importer.return_image(),
                        "image_type" : "Raw", "name" : str(num1), "z_position": -1, "time": -1,}
            self.image_storage[num1] = holder.ImageHolder(**metadata)

    def check_image_length(self):
        """ #checks to see how many images are in the list."""
        return len(self.image_storage)
