"""
Module for developing a class to export files to user specifed location
"""
import os
import json
import csv
import cv2
import numpy.typing as npt
import numpy as np
from skimage import io,color
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import pandas as pd
import matplotlib.cm as cm
from src.core import holder
from src.utils import tools
import random

class ImageExporter:
    """class for saving image from an image holder to a location."""
    def __init__(self,config_file_path = "./config/defaultconfig.json") -> None:

        config = tools.load_config(config_file_path)
        self.image_save_location = config["DataSaveLocation"]

        self.last_saved_location = ""
        self.entity_color_dict = {}

    def save_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """saves an image to folder given an imageholder. 
        Saves to a location dependent on imageholder type. Returns True if saved"""

        img_info = image_holder.return_image_info()
        img = image_holder.return_image()

        image_type = str(img_info["image_type"])
        name = str(img_info["name"])
        #generate names
        save_dir = self.image_save_location+str(image_type)+"Image/"
        self.last_saved_location = save_dir

        save_location = save_dir + name + ".tiff"

        self.make_dir(save_dir)
        return cv2.imwrite(save_location,img)

    def make_dir(self, directory_location: str) -> None:
        """Checks to see if a directory exists. If not, make a directory here. """

        dir_exists = os.path.exists(directory_location)

        if not dir_exists:
            os.makedirs(directory_location)
            return False
        return True

    def save_json(self, property_dictionary: dict = None) -> None:
        """given a dictionary, saves the properties of an image collection to a file."""
        with open(self.last_saved_location+"collection_properties.json","w",encoding="utf-8") as f:
            json.dump(property_dictionary,f,ensure_ascii=False, indent = 4)
        

    def save_3d_image(self, image_holder: holder.ImageHolder) -> npt.ArrayLike:
        """Saves a 3D image as a tiff"""
        img_info = image_holder.return_image_info()
        img = image_holder.return_image()

        image_type = str(img_info["image_type"])
        name = "3Dstack"+str(img_info["name"])
        #generate names
        save_dir = self.image_save_location+str(image_type)+"Image/"
        self.last_saved_location = save_dir

        save_location = save_dir + name + ".tiff"

        self.make_dir(save_dir)
        io.imsave(save_location,img)


    def save_3d_cuts(self,  image_holder: holder.ImageHolder = None, circle_data: npt.ArrayLike = None, classification_data: npt.ArrayLike = None):
        """Saves image holder that represent 3D cuts and draws circles on them.
          Assumes scikit images"""
        title = image_holder.return_image_info()["name"]
        img = image_holder.return_image()
        plt.clf()
        print(img.shape)
        fig1 = plt.imshow(img, cmap='gray')
        self.draw_circles(circle_data,classification_data)


        #save directory:
        #generate names
        img_info = image_holder.return_image_info()
        img = image_holder.return_image()
        image_type = str(img_info["image_type"])
        name = str(img_info["name"])
        save_dir = self.image_save_location+str(image_type)+"Image/"
        self.make_dir(save_dir)
        save_location = save_dir + name + ".tiff"
        plt.title(name)
        plt.savefig(save_location)
        plt.close()
        
        
    def draw_circles(self, particle_data: npt.ArrayLike = None, classification_data: npt.ArrayLike = None) -> None:

        #we can make dashed lines for user input data versus fit data. The color will indicate the entity. So each entity will
        #have a unique color.

        entity_neighbors, particle_entity, particle_input, particle_neighbors = classification_data
        i = False
        if i:
            circles = [Circle((x1,x2), r, edgecolor='red', facecolor='none') for index, x1, x2, r, i, i_mean, i_std in list(particle_data)]
            circle_collection = PatchCollection(circles, match_original=True)
            plt.gca().add_collection(circle_collection)
        else:
            
            colormap = cm.plasma 
            max_value = 20
            neighbor_image = True               

            for entity in entity_neighbors:
                if not entity in self.entity_color_dict:
                    if neighbor_image:

                        neighbor_normalized = min(len(entity_neighbors[entity])/max_value,1)


                        self.entity_color_dict[entity] = facecolor = ( 1-neighbor_normalized, 0.2, neighbor_normalized)
 
                    else:
                        self.entity_color_dict[entity] = (random.random(),random.random(),random.random())




            circle_col = []
            for index, x1, x2, r, i, i_mean, i_std in list(particle_data):
                if index in particle_input:
                    facecolor = "purple"
                    alpha = 0.5
                else:
                    if index in particle_neighbors:
                        neighbors = particle_neighbors[index]
                        number_neighbors = min(len(neighbors)/max_value,1)
                        facecolor = (0.2, 1-number_neighbors, number_neighbors)
                        #facecolor = colormap(min(len(neighbors)/max_value,1))
                        alpha = 0.6
                    else:
                        facecolor = (1,1,0.1)
                        alpha = 0
                
                if index in particle_entity:
                    entity_number = particle_entity[index]
                    edgecolor = self.entity_color_dict[entity_number]
                else:
                    edgecolor = "red"
                    facecolor = "none"
                    alpha = 1

                c1 = Circle((x1,x2), r, edgecolor=edgecolor, facecolor=facecolor, linewidth = 1, alpha = alpha)
                circle_col.append(c1)
            
            circle_collection = PatchCollection(circle_col, match_original=True)
            plt.gca().add_collection(circle_collection)
            




    def save_matplotlib_plot(self, name: str = ""):
        """saves a premade matplotlib_plot"""
        save_dir = self.image_save_location+str("plot")+"Image/"
        self.last_saved_location = save_dir

        save_location = save_dir + name + ".tiff"

        self.make_dir(save_dir)
        plt.savefig(save_location)


    def save_pd_as_csv(self, dataframe = None, name: str = "default"):
        """Saves a pandas dataframe as a CSV file"""
        save_dir = self.image_save_location+str("plot")+"Image/"
        self.last_saved_location = save_dir

        save_location = save_dir + name + ".csv"
        name2 = self.image_save_location.split("/")[-2]+".csv"
        save_location_alt = "./data/AmplitudeSweeps/AmplitudeSweeps_ParticleData/FinalData/"+name2

        ### CHANGE THIS AND MAKE IT PULL FROM A CONFIG
        self.make_dir("./data/AmplitudeSweeps/AmplitudeSweeps_ParticleData/FinalData/")

        dataframe.to_csv(save_location_alt)

