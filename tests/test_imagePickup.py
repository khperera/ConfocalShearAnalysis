"""
Operations for testing module image operations
"""

from src.core import holder, ImageImporter, ImageExporter, ImageSegmentor
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
        image_reader = ImageHolder.ImageHolder()
        image_importer = ImageImporter.ImageImporter()
        imageSaver = ImageExporter.ImageExporter(config_file_path="./config/testingconfig.json")

        image_importer.readLocation(testlocation)
        image_reader.storeImage(image_importer.returnImage(),{"ImageType":"Raw","Name":"TestImage"})
        imageSaver.saveImage(image_reader)

        ImageSegmentor1 = ImageSegmentor.imageSegment()

        ImageSegmentor1.applySegmentation(image_reader)

        imageSaver.saveImage(image_reader)

        assert imageSaver.saveImage(image_reader)


if __name__ == "__main__":
    unittest.main()