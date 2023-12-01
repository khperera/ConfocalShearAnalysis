import sys
#sys.path.insert(0, '../src')
from src.core import ImageHolder, ImageGrabber, ImageSaver
import unittest

class TestImageModule(unittest.TestCase):
    #Checks to see if image is read.
    def test_imageRead(self):
        testlocation = "./tests/TestData/testImage.jpg"
        imageReader = ImageHolder.ImageHolder()
        imageGrabber1 = ImageGrabber.ImageGrabber()


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
        imageGrabber1 = ImageGrabber.ImageGrabber()


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
        imageGrabber1 = ImageGrabber.ImageGrabber()
        imageSaver = ImageSaver.ImageSaver()

        imageGrabber1.readLocation(testlocation)
        imageReader.storeImage(imageGrabber1.returnImage(),{"ImageType":"Raw","Name":"TestImage"})
        self.assertTrue(imageSaver.saveImage(imageReader))

    #False test           
    def test_False(self):
       self.assertFalse( False)
    #True Test
    def test_True(self):
        assert True


if __name__ == "__main__":
    unittest.main()
