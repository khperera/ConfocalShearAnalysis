#import modules
from src.core import collection, viewer
from src.utils import tools
def main():
    list_of_config = ["./config/image_processing_2CdaAmp_40strain.json","./config/image_processing_2CdaAmp_63strain.json",
                      "./config/image_processing_2CdaAmp_200strain.json"]
    load_from_json = False


    for config in list_of_config:
        if not load_from_json:
            image_collection = collection.ImageCollection(config_file_path=config)
            image_collection.load_images()
            image_collection.save_files()
            image_collection.apply_segmentation()
            image_collection.convert_to_3d()
            image_collection.segment_3d_image()
            
            particle_data = image_collection.return_3d_particledata()
            image_collection.save_files()
            #viewer.mayavi_sphere_points3d(particle_data)
            del image_collection
        else:
            #pass
            viewer.load_from_json("./data/ProcessedData/SegmentImage/collection_properties.json")
if __name__ == "__main__":
    main()
    