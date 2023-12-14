#import modules
from src.core import collection, viewer
from src.utils import tools
def main():
    
    image_collection = collection.ImageCollection(config_file_path="./config/image_processing.json")
    image_collection.load_images()
    image_collection.save_files()
    image_collection.apply_segmentation()
    image_collection.convert_to_3d()
    image_collection.segment_3d_image()
    particle_data = image_collection.return_3d_particledata()
    image_collection.save_files()
    viewer.plotfigure(particle_data)
if __name__ == "__main__":
    main()
    