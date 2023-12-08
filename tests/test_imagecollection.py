"""
Unit tests for testing image collection methods/ operations.
"""
import unittest
from src.core import collection


class TestImageCollectionModule(unittest.TestCase):
    """Checks to see if collector modifications are read.
    test to see if 2 images in a folder could be loaded"""
    def test_imagecollection_loading(self):
        """
        Sees if an image collection can load set images in directory.
        """
        image_collection = collection.ImageCollection(image_location="./tests/TestData/",
                                                    config_file_path="./config/testingconfig.json")
        image_collection.load_images()
        num_images = image_collection.check_image_length()
        if num_images == 5:
            assert True
        else:
            assert False


    def test_adding_emptyholders(self):
        """test to see if we can add image holders, and would it hold right amount"""
        image_collection = collection.ImageCollection(image_location="./tests/TestData/",
                                config_file_path="./config/testingconfig.json")
        finnum_images = 11
        for _ in range(0,finnum_images):
            image_collection.add_image_holder()
        num_images =  image_collection.check_image_length()

        if finnum_images == num_images:
            assert True
        else:
            assert False

    def test_collector_saving(self):
        """checks to see if saving a collection of files works."""
        image_collection = collection.ImageCollection(image_location="./tests/TestData/",
                                                    config_file_path="./config/testingconfig.json")
        image_collection.load_images()
        assert image_collection.save_files()



    def test_collector_segmentandsave(self):
        """#checks to see if segmenting a collection of files works."""
        image_collection = collection.ImageCollection(image_location="./tests/TestData/",
                                                    config_file_path="./config/testingconfig.json")
        image_collection.load_images()
        image_collection.apply_segmentation()
        image_collection.save_files()

if __name__ == "__main__":
    unittest.main()
