"""
Operations for testing module image operations
"""
import unittest
from src.core import holder, importer, exporter, segmentor

class TestImageModule(unittest.TestCase):
    """Test image classification module"""
    def test_imageread(self):
        """Test to see if an image can be read."""
        testlocation = "./tests/TestData/testImage.jpg"
        image_storage = holder.ImageHolder()
        image_import = importer.ImageImporter()
        image_import.read_location(testlocation)
        image_storage.store_image(image_import.return_image())
        size = image_storage.return_image_size()

        if size[0] > 1:
            assert True
        else:
            assert False

    def test_imageread_badlocation(self):
        """Test for invalid image location. 
            If finds a bad image, returns true. Checks x dimension of image"""
        testlocation = "./tests/TestData/NOTREALLOCATION.jpg"
        image_storage = holder.ImageHolder()
        image_import = importer.ImageImporter()


        image_import.read_location(testlocation)
        image_storage.store_image(image_import.return_image())
        size = image_storage.return_image_size()
        #print(size)
        if size[0] > 1:
            assert False
        else:
            assert True

    def test_imagesave(self):
        """Verification that image can be saved"""
        testlocation = "./tests/TestData/testImage.jpg"
        image_storage = holder.ImageHolder()
        image_import = importer.ImageImporter()
        image_export = exporter.ImageExporter(config_file_path="./config/testingconfig.json")

        image_import.read_location(testlocation)
        image_storage.store_image(image_import.return_image(),
                                    {"ImageType":"Raw","Name":"TestImage"})
        self.assertTrue(image_export.save_image(image_storage))


    def test_imagesegmentor(self):
        """Validation for image segmenting and saving"""
        testlocation = "./tests/TestData/testImage.jpg"
        image_storage = holder.ImageHolder()
        image_importer = importer.ImageImporter()
        image_export = exporter.ImageExporter(config_file_path="./config/testingconfig.json")

        image_importer.read_location(testlocation)
        image_storage.store_image(image_importer.return_image(),
                                  {"ImageType":"Raw","Name":"TestImage"})
        image_export.save_image(image_storage)

        image_segment= segmentor.ImageSegment()

        image_segment.apply_segmentation(image_storage)

        image_export.save_image(image_storage)

        assert image_export.save_image(image_storage)


if __name__ == "__main__":
    unittest.main()
