"""
Unit tests for testing image collection methods/ operations.
"""
import unittest
from src.core import holder, importer, exporter, segmentor, collection


class TestImageCollectionModule(unittest.TestCase):
    """Checks to see if collector modifications are read.
    test to see if 2 images in a folder could be loaded"""
    def test_imagecollection_loading(self):
        """
        """
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/", config_file_path="./config/testingconfig.json")
        imageCollection1.loadImages()
        numberOfImages = imageCollection1.checkImageLength()
        if numberOfImages == 5:
            assert True
        else:
            assert False

    #test to see if we can add image holders, and would it hold right amount
    def test_addingImageHolders(self):
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/", config_file_path="./config/testingconfig.json")
        numberofImages = 11
        for val1 in range(0,numberofImages):
            imageCollection1.addImageHolder()
        finalnumberofImages =  imageCollection1.checkImageLength()

        if finalnumberofImages == numberofImages:
            assert True
        else:
            assert False

    #checks to see if saving a collection of files works.
    def test_collector_saving(self):
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/", config_file_path="./config/testingconfig.json")
        imageCollection1.loadImages()
        numberOfImages = imageCollection1.checkImageLength()
        imageCollection1.saveFiles()


    #checks to see if segmenting a collection of files works.
    def test_collector_segmentandsave(self):
        pass
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/", config_file_path="./config/testingconfig.json")
        imageCollection1.loadImages()
        numberOfImages = imageCollection1.checkImageLength()
        imageCollection1.applySegmentation()
        imageCollection1.saveFiles()
    
    
    
    #False test           
    def test_False(self):
       self.assertFalse( False)
    #True Test
    def test_True(self):
        assert True


if __name__ == "__main__":
    unittest.main()
