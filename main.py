#import modules
from src.core import collection
from src.utils import tools
def main():
    image_collection = collection.ImageCollection(config_file_path="./config/image_processing.json")
    image_collection.load_images()
    image_collection.save_files()
    image_collection.apply_segmentation()
    image_collection.save_files()
if __name__ == "__main__":
    main()
    