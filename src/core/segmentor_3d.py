"""
Class that is used to segment 3D images. Must feed in unblurred images.
"""
import math
import numpy as np
import numpy.typing as npt
from skimage import filters
from skimage import img_as_ubyte
from skimage.transform import rescale as scale1
from sklearn.neighbors import BallTree
import skimage as ski
import scipy 
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.utils.particlefinder_nogpu import MultiscaleBlobFinder, OctaveBlobFinder, get_deconv_kernel
from src.utils import rescale
from src.core import holder, exporter
from src.utils import tools


class ImageSegment():

    """
    #this will be a class that is init once and will take in files indivitually to do edits.
    #It interfaces with the image holder app by taking it in, 
    applying relevant segmenting information,
    #then overwriting with segmented image.
    #segmentation instruction passed via config file. 
    Should have option to call segmentation without config?
    """

    def __init__(self,config_file_path: str = "./config/testingconfig.json") -> None:
        #load in configuration files.
        config = tools.load_config(config_file_path=config_file_path)

        self.display_original_img = config["segment_config"]["display_original_img"]
        #create default image.
        self.img = np.zeros((1,1,3), dtype=np.uint8)
        self.img_original = np.zeros((1,1,3), dtype=np.uint8)
        self.img_shape = None
        self.particle_centers = None
        self.kernel = None
        self.config_file_path= config_file_path
        self.particle_addition_save_location = config["3DParticleData"]


        #image segmentation variables.
        self.alpha = 0.2
        self.beta = 1.2
        self.max_radius = 3
        self.balltree = None
        self.entity_indicator = 0

        #particledata_storage
        #stores the neighbors of each particle
        self.particle_dict = {}
        #stores the entity of each particle
        self.particle_entity = {}
        #entitystorage, stores the particles contained by each entity.
        self.entity_dict = {}
        #the ignore list. Don't consider these particles or draw them.
        self.ignore_list = []
        #entity_neighborlist
        self.entity_neighbors = {}
        #cluster data (r,x,y,z,neighbors)
        self.entity_info = []

        #input particle list. List particles to be colored when drawn.
        self.particle_in = []

####################################################################################################
#main function. callable by other functions, classes


    #applies all the segmenting tasks needed to an image holder.
    #Takes in an imageholder
    def apply_segmentation(self, image_holder: holder.ImageHolder = holder.ImageHolder()):
        """Given the config file, applies segmentation across all images in stack"""

        self.img = image_holder.return_image()
        self.squish_axis()
        self.find_particle_centers()
        self.load_particle_data()
        self.sample_spheres(samples = 1000)
        self.save_cuts(image_type="3D_cut_priorfilter")
        #particle_centers changes shape after prev Operation
        self.filter_particles()
        self.generate_entity_data()
        self.find_entity_neighbors()
        self.filter_particles_ignorelist()
        self.generate_cluster_information()
        dictionary_fit = image_holder.return_fit_dictionary()
        dictionary_fit["position"] = np.ndarray.tolist(self.particle_centers)

    def return_particledata(self):
        """returns the particle center data"""
        return self.particle_centers
    
    def load_particle_data(self):
        """Loads particle data from CSV. In form, x, y, z, r, i """
        input_array = np.genfromtxt(self.particle_addition_save_location, delimiter=",")
        #print(input_array)
        #print(input_array.shape)
        rows = input_array.shape[0]
        max_index = np.array(self.particle_centers).max(axis=0)[0]
        index_new = np.arange(max_index+1, max_index+1+rows)
        self.particle_in = index_new
        new_particle_data = np.column_stack((index_new,input_array))
        #print(new_particle_data, new_particle_data.shape)
        self.particle_centers = np.concatenate((self.particle_centers, new_particle_data), axis = 0)

        

    def export_radius_neighbor_data(self):
        """Exports a list of tuples to CSV. Radius vs number of neighbbors. Also exports to different
        files the average, mediam radius and average, median number of neighbors"""
        pass

    def generate_cluster_information(self):
        """Generates a list of radius, weighted x,y,z, vs number of neighborsf for each cluster"""
        #contains the weighted radius, weighted xyz, and number of neighbors for each entity


        for entity, value in self.entity_dict.items():
            particle_list = value
            list_of_radii = []
            list_of_positions = []
            weights = []
            weighted_position = (0,0,0)
            sum_radius2 = 0

            for particle in particle_list:
                rows_with_value = self.particle_centers[self.particle_centers[:, 0] == particle] 
                radius = rows_with_value[0,4]
                coordinates = rows_with_value[0,1:4].reshape(1,-1)
                list_of_radii.append(radius)
                list_of_positions.append(coordinates)

                sum_radius2 = sum_radius2+radius**2
                weights.append(radius**2)
            weights = np.array(weights)
            weights = weights/np.sum(weights)

            radius_weighted = np.sum(np.array(list_of_radii)*weights)
            
            x_ = 0
            y_ = 0 
            z_ = 0
            for count, coordinate in enumerate(list_of_positions):
                x_ = coordinates[0]*weights[count] + x_
                y_ = coordinates[1]*weights[count] + y_
                z_ = coordinates[2]*weights[count] + z_

            neighbors = len(self.entity_neighbors[entity])

            self.entity_info.append([radius_weighted,x_,y_,z_, neighbors])


            





            
            



####################################################################################################
#filters
    

    

    def create_kernel(self) -> None:
        """Creates deconvulusion kernel"""
        self.kernel = get_deconv_kernel(self.img, pxZ = 1, pxX=1)

    def find_particle_centers(self)-> None:
        """Uses prytrack to find particle centers"""
        self.convert_to_scikit_image()
        self.create_kernel()
        finder = MultiscaleBlobFinder(self.img.shape, Octave0=False, nbOctaves=4)
        centers = finder(self.img, maxedge = -1,removeOverlap=False,deconvKernel=self.kernel)
        rescale_intensity = True
        if rescale_intensity:
            s = rescale.radius2sigma(centers[:,-2], dim=3)
            bonds, dists = rescale.get_bonds(positions=centers[:,:-2], radii=centers[:,-2], maxdist=3.0)
            brights1 = rescale.solve_intensities(s, bonds, dists, centers[:,-1])
            radii1 = rescale.global_rescale_intensity(s, bonds, dists, brights1)
        else:
            radii1 = centers[:,-2]
        # Assuming centers is a NumPy array
        length = len(centers[:,0])  # Get the length of the first column of centers

        # Generate an array of indices from 0 to length-1
        index_array = np.arange(length)
        particledata = list(zip(index_array, centers[:,0], centers[:,1], centers[:,2], radii1, centers[:,-1]))

        self.particle_centers = particledata




        return particledata


    def save_cuts(self, x_divs = 50, y_divs = 50, z_divs = 50, image_type = "3D_cut"):
        """Call after segmenting and getting particle data
        Saves cuts and displays circles on them, saves to holder then exports"""
        img_size = self.img.shape
        print(img_size)
        x_size = img_size[2]/x_divs
        y_size = img_size[1]/y_divs
        z_size = img_size[0]/z_divs

        saver = exporter.ImageExporter(config_file_path=self.config_file_path)
        #need to find particle that are in plane with the cut and find radius that is in plane.
        i = 1
        #x_cuts
        for x_div in range(x_divs):
            pixel_dim = int(x_div*x_size)
            yz_cut = self.img[:,:,pixel_dim].copy()
            name = "yzcut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(x=pixel_dim)
            #print(circles_in_dim.shape, "shapein")
            holder_yz = holder.ImageHolder(yz_cut,image_type=image_type,name= name)
            saver.save_3d_cuts(holder_yz,circles_in_dim, [self.entity_neighbors,self.particle_entity,self.particle_in, self.particle_dict])

        for y_div in range(y_divs):
            pixel_dim = int(y_div*y_size)
            xz_cut = self.img[:,pixel_dim].copy()
            name = "xzcut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(y=pixel_dim)
            holder_xz = holder.ImageHolder(xz_cut,image_type=image_type,name= name)
            saver.save_3d_cuts(holder_xz,circles_in_dim, [self.entity_neighbors,self.particle_entity,self.particle_in, self.particle_dict])

        for z_div in range(z_divs):
            pixel_dim = int(z_div*z_size)
            xy_cut = self.img[pixel_dim].copy()
            name = "xycut_" + str(pixel_dim) + "pixels"
            circles_in_dim = self.circles_in_dimension(z=pixel_dim)
            holder_xy = holder.ImageHolder(xy_cut,image_type=image_type,name= name)
            saver.save_3d_cuts(holder_xy,circles_in_dim, [self.entity_neighbors,self.particle_entity,self.particle_in, self.particle_dict])
        
        self.plot_histograms()
        saver.save_matplotlib_plot("Intensity plots_postfilter")

        

    def squish_axis(self, x_squish: float = 0.75, y_squish: float = 0.75, z_squish: float =1):
        self.img = scale1(self.img, (z_squish,x_squish,y_squish), anti_aliasing = True )


    def plot_histograms(self):
        """plot particle data"""
        all_indicies = np.arange(len(self.particle_centers))
        rows_to_keep = np.setdiff1d(all_indicies,self.ignore_list)

        filtered_particle_centers = self.particle_centers[rows_to_keep]
        edgeintensity = [item[-3] for item in filtered_particle_centers]
        radii = [item[-4] for item in filtered_particle_centers]
        ratioedgeradii = [item[-3]/item[-4] for item in filtered_particle_centers]
        intensitymean =  [item[-2] for item in filtered_particle_centers]
        intensitystd =  [item[-1] for item in filtered_particle_centers]


        fig, axs = plt.subplots(2,3, constrained_layout = True)

        ax1 = axs[0,0]
        ax2 = axs[0,1]
        ax3 = axs[1,0]
        ax4 = axs[1,1]
        ax5 = axs[0,2]



        # Now plot the histogram
        ax1.hist2d(radii, edgeintensity, bins = 100, norm= mpl.colors.LogNorm())  # Adjust bins as needed

        ax1.set_title('Radii vs intensity histo')
        


        ax2.hist(ratioedgeradii, bins = 100)

        ax2.set_title('intensity/radii')
        ax2.set_xlabel("Intensity/Radii")
        ax3.hist2d(radii, intensitymean, bins = 100, norm= mpl.colors.LogNorm())  # Adjust bins as needed
        ax3.set_xlabel("Radii")
        ax3.set_ylabel("MeanIntensity")


        ax3.set_title('Intensity vs radii histo')

        ax4.hist2d(radii, intensitystd, bins = 100, norm= mpl.colors.LogNorm())  # Adjust bins as needed
        ax4.set_xlabel("Radii")
        ax4.set_ylabel("IntensityDeviation")

        ax4.set_title('Intensitystd vs radii histo')

        ax5.hist2d(intensitymean, intensitystd, bins = 100, norm= mpl.colors.LogNorm())  # Adjust bins as needed
        ax5.set_xlabel("IntensityMean")
        ax5.set_ylabel("IntensityStd")
        ax5.set_title('Intensity Deviation vs Average Intensity')
        





####################################################################################################
#utils


    def convert_to_single_channel(self, channel = 1):
        """
        #checks to see if an image is of the type uint8. Chooses the correct channel as well
        #channel 0,1,2 is b,g,r
        """
        self.img = np.uint8(cv2.split(self.img)[channel])

    def convert_to_scikit_image(self):
        """Converts an opencv image to scikit-image version, single channel"""
        ski.util.img_as_float(self.img)

    def convert_to_opencv_image(self):
        """Converts a scikit image to opencv image, single channel"""
        self.img = img_as_ubyte(self.img)

    def circles_in_dimension(self,x: int = None,y: int = None,z: int = None)-> npt.ArrayLike:
        """Returns the list of circles that are in the dimension and 
        gives effect radius at that plane"""
        positions = np.array(self.particle_centers)
        print(self.particle_centers.shape, "shape")
        if x is not None:
            
            x_values = positions[:,1]
            r_values = positions[:, 4]
            filtered_positions = positions[np.abs(x_values-x) - r_values <= 0]
            #pythagorean theorem
            filtered_positions[:,4] = np.sqrt(filtered_positions[:,4]**2 - (filtered_positions[:,1]-x)**2)
            
            return np.delete(filtered_positions,1, axis = 1)
        if y is not None:
            
            y_values = positions[:,2]
            r_values = positions[:, 4]
            filtered_positions = positions[np.abs(y_values-y) - r_values <= 0]
            filtered_positions[:,4] = np.sqrt(filtered_positions[:,4]**2 - (filtered_positions[:,2]-y)**2)
            return np.delete(filtered_positions,2, axis = 1)
        if z is not None:
            
            z_values = positions[:,3]
            r_values = positions[:, 4]
            filtered_positions = positions[np.abs(z_values-z) - r_values <= 0]
            filtered_positions[:,4] = np.sqrt(filtered_positions[:,4]**2 - (filtered_positions[:,3]-z)**2)
            return np.delete(filtered_positions,3, axis = 1)

        return positions
    
###############################################################
#Filtering data/methodologies
###################################################



    def generate_entity_data(self):
        """Uses balltree to determine neighbors, and circle subfiltering."""
        spatial_data = self.particle_centers[:,0:4]


        self.balltree = BallTree(spatial_data[:,1:4], metric = "euclidean")

        self.max_radius = np.max(self.particle_centers[:,4])

        #implement a recursive algorithm. Need to find all particles in a entity or neighbor.
        #three actions, add particle to remove list, add particle to an entity, add particle as a neighbor.
        #start by seeing if a particle has been identified before. If so, continue.
        #Check all nearest neighbors w/in radius of a particle. 
        #if neighbor is fully within, add to ignorelist
        #if particle is in entity range, add it to current entity. Then activate recursive entity track.
            #See if that subparticle has neighbors and entities. Activate recursive. The break condition is if all particles seen.
        #If particle is in neighbor range, add to system neighbor.

        #at end, algorithm to see the neighboring entities.

        for particle in spatial_data:
            count, x, y, z = particle
            #Check to see if we have identified the particle already.  this can remain. Other stuff must go
            current_entity = self.entity_indicator
            if count in self.particle_dict:
                continue
   
            self.identify_entities(particle=count, entity_number=current_entity)
            self.entity_indicator = self.entity_indicator + 1


    def identify_entities(self, particle = 0, entity_number = 0):
        """A recursive function that identifies particles"""
        #make particle dict global
        if particle in self.particle_dict:
            return
        else:
            self.particle_dict[particle] = []
            self.particle_entity[particle] = entity_number
        
        if entity_number not in self.entity_dict:
            self.entity_dict[entity_number] = []
            
        #print(particle)
        #balltree must be made global
        rows_with_value = self.particle_centers[self.particle_centers[:, 0] == particle]   
        radius = rows_with_value[0,4]
        coordinates = rows_with_value[0,1:4].reshape(1,-1)
        #print(radius, coordinates)
        array_tech = self.balltree.query_radius(coordinates, r = self.max_radius*2)

        for row in array_tech[0][1:]:
            
            
            radius_2 = self.particle_centers[row,4]
            coordinates_2 = self.particle_centers[row,1:4]
            #print(coordinates_2, radius_2)
            distance = scipy.spatial.distance.euclidean(coordinates.flatten(), coordinates_2) 
            particle_index = self.particle_centers[row,0]

            #check to see if fully contained, then delete if true.
            if (radius > radius_2+distance):
                
                self.ignore_list.append(particle_index)
                continue

            if (radius >= (distance - radius_2*self.alpha) or radius_2 >= (distance - radius*self.alpha)):
                self.entity_dict[entity_number].append(particle_index)
                self.identify_entities(particle=particle_index, entity_number=entity_number)
                continue

            if ((self.beta*(radius + radius_2)) > (distance)):
                self.particle_dict[particle].append(particle_index)
                continue
   
            

    def find_entity_neighbors(self):
        """Iterates through all the entities and find their neighbors"""

        for entity in self.entity_dict:
            
            unique_neighbors = []
            for particle in self.entity_dict[entity]:
                
                neighbors = self.particle_dict[particle]

                for neighbor in neighbors:
                    neighboring_entity = self.particle_entity[neighbor]

                    if ((not neighboring_entity in unique_neighbors) and (not neighboring_entity is entity)):
                        unique_neighbors.append(neighboring_entity)
            
            self.entity_neighbors[entity] = unique_neighbors
            print(len(unique_neighbors))
            
        













###############################################################
#Filtering data/methodologies
###################################################
    def generate_sampling_space(self, samples: int = 50):
        """Sub-sample particles to clean up the particles. Takes a list of particles in. Finds them in the image.
        Returns a sampling space of r,theta,phi.
            """

        rng = np.random.default_rng()
        
        r = np.log10(rng.random(size = samples)*10)
        theta = rng.random(size = samples)*2*np.pi
        phi = rng.random(size = samples)*np.pi

        return np.column_stack((r,theta,phi))
    
    def sample_spheres(self, samples: int = 50):
        """Returns the standard deviation and mean of the sphere sampling. 
        Takes in a list of particles. Returns the particles concat with their avg and stdev intesnity """

        overall_statistics = []
        for particle in self.particle_centers:
            index,x,y,z,r,i = particle
            sphere_sampling = self.generate_sampling_space(samples= samples)
            particle_data = []
            for sampling in sphere_sampling:
                r_nondim, theta, phi = sampling
                r_dim = r_nondim*r
                #finding the shift from center
                x_s = r_dim*np.cos(theta)*np.sin(phi)
                y_s = r_dim*np.sin(theta)*np.sin(phi)
                z_s = r_dim*np.cos(phi)

                #Calculating the access point of the image.
                z_max, y_max, x_max = self.img.shape

                x_access = max(min(math.trunc(x + x_s),x_max-1),0)
                y_access = max(min(math.trunc(y + y_s), y_max-1),0)
                z_access = max(min(math.trunc(z + z_s), z_max-1),0)

                #verification that everything works. REMOVE. Works by turning the sampled area to 0. If see speckles in image, then works.
                #self.img[z_access,y_access,x_access] = 0

                val = self.img[z_access,y_access,x_access]

                particle_data.append(val)
            
            particle_data = np.array(particle_data)
            intensity_std = np.std(particle_data)
            intensity_mean = np.mean(particle_data)
            overall_statistics.append((intensity_mean,intensity_std))
        self.particle_centers = np.column_stack((self.particle_centers,overall_statistics))

        
        saver = exporter.ImageExporter(config_file_path=self.config_file_path)
        self.plot_histograms()
        saver.save_matplotlib_plot("Intensity plots_prefilter")


    def filter_particles(self):
        """Uses various metrics to filter the particles by radius, edge intensity, average intensity, intensity deviation"""
        radius_threshold_high = 100
        radius_threshold_low = 0
        std_dev_thresh_high = 0.23
        mean_thresh_low = 0.3

        #filtered array is z,y,x, r, i, intensity_means, intensity_std

        filtered_array = self.particle_centers

        #radius filters
        filtered_array = filtered_array[(filtered_array[:,4]>radius_threshold_low) & (filtered_array[:,4]<radius_threshold_high) ]

        #intensity filter
        filtered_array = filtered_array[(filtered_array[:,6]>mean_thresh_low) & (filtered_array[:,7]<std_dev_thresh_high) ]
        self.particle_centers = np.copy(filtered_array)

    def filter_particles_ignorelist(self):
        """Filters particles based on the ignorelist generated by by the nearest neighbor analysis"""

        # Assuming self.particle_centers is your 2D array and rows_to_remove is the list of row indices to remove
        values_to_remove = self.ignore_list  # Example indices of rows to remove

        mask = ~np.isin(self.particle_centers[:, 0], values_to_remove)

        self.particle_centers = self.particle_centers[mask]


