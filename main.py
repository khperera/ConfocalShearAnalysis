#import modules
from src.core import collection, viewer
from src.utils import tools
def main():
    list_of_config = [
                     "./config/image_processing_1CdaAmp_1strain.json",
"./config/image_processing_1CdaAmp_10strain.json",
"./config/image_processing_1CdaAmp_20strain.json",
"./config/image_processing_1CdaAmp_30strain.json",
"./config/image_processing_1CdaAmp_40strain.json",
"./config/image_processing_1CdaAmp_50strain.json",
"./config/image_processing_1CdaAmp_63strain.json",
"./config/image_processing_1CdaAmp_80strain.json",
"./config/image_processing_1CdaAmp_100strain.json",
"./config/image_processing_1CdaAmp_200strain.json",
"./config/image_processing_1CdaAmp_630strain.json",
"./config/image_processing_1CdaAmp_1000strain.json",
"./config/image_processing_1CdaAmp_xyz.json",
"./config/image_processing_2CdaAmp_1strain.json",
"./config/image_processing_2CdaAmp_10strain.json",
"./config/image_processing_2CdaAmp_20strain.json",
"./config/image_processing_2CdaAmp_30strain.json",
"./config/image_processing_2CdaAmp_40strain.json",
"./config/image_processing_2CdaAmp_50strain.json",
"./config/image_processing_2CdaAmp_63strain.json",
"./config/image_processing_2CdaAmp_80strain.json",
"./config/image_processing_2CdaAmp_100strain.json",
"./config/image_processing_2CdaAmp_200strain.json",
"./config/image_processing_2CdaAmp_630strain.json",
"./config/image_processing_2CdaAmp_1000strain.json",
"./config/image_processing_2CdaAmp_xyz.json",
"./config/image_processing_4CdaAmp_1strain.json",
"./config/image_processing_4CdaAmp_10strain.json",
"./config/image_processing_4CdaAmp_20strain.json",
"./config/image_processing_4CdaAmp_30strain.json",
"./config/image_processing_4CdaAmp_40strain.json",
"./config/image_processing_4CdaAmp_50strain.json",
"./config/image_processing_4CdaAmp_63strain.json",
"./config/image_processing_4CdaAmp_80strain.json",
"./config/image_processing_4CdaAmp_100strain.json",
"./config/image_processing_4CdaAmp_200strain.json",
"./config/image_processing_4CdaAmp_630strain.json",
"./config/image_processing_4CdaAmp_xyz.json"

                      
                      ]
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
    