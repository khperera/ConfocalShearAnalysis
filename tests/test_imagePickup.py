import sys
#sys.path.insert(0, '../src')
from src.core import ImageHolder, ImageImporter, ImageExporter, ImageSegmentor, ImageCollection
import unittest

class TestImageModule(unittest.TestCase):
    #Checks to see if image is read.
    def test_imageRead(self):
        testlocation = "./tests/TestData/testImage.jpg"
        imageReader = ImageHolder.ImageHolder()
        imageGrabber1 = ImageImporter.ImageImporter()


        imageGrabber1.readLocation(testlocation)
        imageReader.storeImage(imageGrabber1.returnImage())
        size = imageReader.returnImageSize()
        #print(size)
        if size[0] > 1:
            assert True
        else:
            assert False

    #Test for invalid image location. If finds a bad image, returns true. Checks x dimension of image
    def test_imageRead_badLocation(self):
        testlocation = "./tests/TestData/NOTREALLOCATION.jpg"
        imageReader = ImageHolder.ImageHolder()
        imageGrabber1 = ImageImporter.ImageImporter()


        imageGrabber1.readLocation(testlocation)
        imageReader.storeImage(imageGrabber1.returnImage())
        size = imageReader.returnImageSize()
        #print(size)
        if size[0] > 1:
            assert False
        else:
            assert True

    def test_imageSave(self):
        testlocation = "./tests/TestData/testImage.jpg"
        imageReader = ImageHolder.ImageHolder()
        imageGrabber1 = ImageImporter.ImageImporter()
        imageSaver = ImageExporter.ImageExporter(config_file_path="./config/testingconfig.json")

        imageGrabber1.readLocation(testlocation)
        imageReader.storeImage(imageGrabber1.returnImage(),{"ImageType":"Raw","Name":"TestImage"})
        self.assertTrue(imageSaver.saveImage(imageReader))

    #checks if the image segmentor initializes
    def test_imageSegmentorInit(self):
        testlocation = "./tests/TestData/testImage.jpg"
        imageReader = ImageHolder.ImageHolder()
        imageGrabber1 = ImageImporter.ImageImporter()
        imageSaver = ImageExporter.ImageExporter()

        imageGrabber1.readLocation(testlocation)
        imageReader.storeImage(imageGrabber1.returnImage(),{"ImageType":"Raw","Name":"TestImage"})
        imageSaver.saveImage(imageReader)

        ImageSegmentor1 = ImageSegmentor.imageSegment()

        ImageSegmentor1.applySegmentation(imageReader)
        
        imageSaver.saveImage(imageReader)
        
        assert imageSaver.saveImage(imageReader)

    #test to see if 2 images in a folder could be loaded
    def test_imageCollectionLoading(self):
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/")
        imageCollection1.loadImages()
        numberOfImages = imageCollection1.checkImageLength()
        if numberOfImages == 2:
            assert True
        else:
            assert False

    #test to see if we can add image holders, and would it hold right amount
    def test_addingImageHolders(self):
        imageCollection1 = ImageCollection.ImageCollection(imageLocation="./tests/TestData/")
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
    
    #False test           
    def test_False(self):
       self.assertFalse( False)
    #True Test
    def test_True(self):
        assert True


if __name__ == "__main__":
    unittest.main()
