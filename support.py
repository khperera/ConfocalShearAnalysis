from support2 import *
#weka segmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import skimage
from skimage.io import imread
import warnings
warnings.simplefilter('ignore', np.RankWarning)
import joblib
import statistics
import cv2
from skimage.morphology import skeletonize
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import img_as_ubyte
import math
import networkx as nx
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy.optimize import leastsq
from sklearn.cluster import DBSCAN
from scipy.optimize import leastsq
from scipy.ndimage import distance_transform_edt, sobel

###################################################

#general functions
###################################################
def fit_circle(points):
    def calc_distance(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2(c):
        Ri = calc_distance(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = leastsq(f_2, center_estimate)
    return center[0], center[1], calc_distance(*center).mean()



###################################################

#classes
###################################################

#calls in an image and segments it. Holds 1 image.
class imageAnalysis:
    def __init__(self, imageLocation):
        self.imageLocation = imageLocation
        self.img = ""
        self.img1 = ""
        #holds collection of thresholded images
        self.threshImages = []
        self.blobsInfo = []
        self.massFract = 0
    def __del__(self):
        del self.img
        del self.img1
        for i in self.threshImages:
            del i
        for i in self.blobsInfo:
            del i
    #opens image
    def openImage(self):

        self.img = imread(self.imageLocation)
        self.image1 = cv2.imread(self.imageLocation,1)
        self.b,self.g,self.red1 = cv2.split(self.image1)




    #this is typical watershed method
    #image are too low resolution to get much better than this
    def segmentImage1(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)

        #ret, thresh = cv2.threshold(red,0,255,cv2.THRESH_BINARY +
        #                                            cv2.THRESH_OTSU)

        #cv2.imshow('bilateral IMage',red)
        #cv2.imshow('blur IMage',red2)
        #cv2.imshow('OG IMage',red1)
        #cv2.imshow('bilateral IMage 60',redblur1)
        #cv2.imshow('bilateral 90',redblur2)


        l = [10,20,50,100,200]
        n = [100,150,200,300,400,500]

        for b in l:


            for thing in n:
                if thing>b:
                    #name1 = "canny edge" + str(b)+str(thing)
                    #edges = cv2.Canny(red,b,thing)
                    #cv2.imshow(name1, edges)
                    b = thing


        edges = cv2.Canny(red1,350,400)
        #cv2.imshow('Canny Edges', edges)

        #red = cv2.subtract(red,edges)

        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)


        #cv2.imshow('segment',thresh)


        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)



        #for test case

        #opening2 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


        #dist_transform2 = cv2.distanceTransform(opening,cv2.DIST_L2,3)


        #ret, sure_fg2 = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        #sure_fg2 = sure_fg2-edges


        #cv2.imshow('distancetransform2',sure_fg2)












        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img2,markers)
        self.image1[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]
        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        #cv2.imshow('blu',self.image1)

        self.threshImages.append(finalParticels1)






        print("done")



        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imshow('blu1',self.image1)
    def segmentImage3(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)

        #ret, thresh = cv2.threshold(red,0,255,cv2.THRESH_BINARY +
        #                                            cv2.THRESH_OTSU)

        #cv2.imshow('bilateral IMage',red)
        #cv2.imshow('blur IMage',red2)
        #cv2.imshow('OG IMage',red1)
        #cv2.imshow('bilateral IMage 60',redblur1)
        #cv2.imshow('bilateral 90',redblur2)



        edges = cv2.Canny(red1,350,400)
        #cv2.imshow('Canny Edges', edges)

        #red = cv2.subtract(red,edges)

        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)


        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)







        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Finding elongated blobs and apply erosion to split them
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1




        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.

        # Create an image to show blobs before erosion.
        pre_erosion = np.zeros_like(sure_fg)

        # Create an empty image to store the results of all the erosions.
        erosion_result = np.zeros_like(sure_fg)
        # Set a threshold for circularity.
        circularity_threshold = 0.80  # Adjust this value as needed.





        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]


            # Only consider blobs above a certain size.
            if area < size_threshold:
                continue


            blob = np.where(labels == i, 255, 0).astype('uint8')

            # Find contours of the blob.
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume there is only one contour (the blob itself).
            contour = contours[0]

            # Calculate area and perimeter (arc length).
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                perimeter = 1
                area = 0.01

            # Calculate circularity.
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < circularity_threshold:

                # Save the blobs to be eroded in pre_erosion image.
                pre_erosion = cv2.bitwise_or(pre_erosion, blob)


                # Erode the blob if its circularity is below the threshold.
                blob = cv2.erode(blob, erosion_kernel, iterations = 1)
                # Remove the original blob from the markers.
                markers[labels == i] = 0

                # Update the erosion_result image with the eroded blob.
                erosion_result = cv2.bitwise_or(erosion_result, blob)

                # Run connectedComponents on the eroded blob.
                num, labels_blob = cv2.connectedComponents(blob)

                # Assign a unique label to each blob resulting from the erosion.
                for j in range(1, num):
                    markers[labels_blob == j] = new_label
                    new_label += 1


        cv2.imshow('Sure FG', sure_fg)
        cv2.imshow('Final Erosion Result', erosion_result)
        cv2.imshow('Blobs before Erosion', pre_erosion)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img2, markers)
        self.image1[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]




        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        cv2.imshow('blu',finalParticels1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)


        print("done")





    def segmentImage5(self):
        red1 = self.red1
        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)

        l = [10,20,50,100,200]
        n = [100,150,200,300,400,500]
        for b in l:
            for thing in n:
                if thing>b:
                    b = thing

        thresh = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1

        # Copy the sure foreground image.
        binary = sure_fg.copy()

        # Create an empty image to store the pre-erosion blobs.
        pre_erosion = np.zeros_like(sure_fg)

        # Set a threshold for circularity.
        circularity_threshold = 0.80  # Adjust this value as needed.

        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5

        # Run connectedComponents on the sure_fg image to get the initial blobs to process.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
        blobs_to_process = list(range(1, num_labels))

        # Store the original markers image for reference.
        original_markers = markers.copy()

        # Loop until no more blobs need to be processed.
        while blobs_to_process:

            # Take a blob from the queue.
            original_blob_index = blobs_to_process.pop(0)

            # Run connectedComponents on the binary image to get the updated blobs and their stats.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Find the new index of the blob in labels
            new_blob_index = np.where(labels == original_blob_index)[0]

            if not len(new_blob_index) > 0:
                continue
            i = new_blob_index
            # Get the statistics of this blob.
            stats_blob = stats[i]

            # Calculate the aspect ratio of this blob.
            aspect_ratio_blob = float(stats_blob[cv2.CC_STAT_WIDTH]) / float(stats_blob[cv2.CC_STAT_HEIGHT])

            # Calculate the area of this blob.
            area_blob = stats_blob[cv2.CC_STAT_AREA]

            # Check if the aspect ratio is above the threshold.
            if aspect_ratio_blob > aspect_ratio_threshold:

                # Get the region of the binary image that contains the blob.
                blob_region = binary[labels == i]

                # Erode the blob in the binary image.
                eroded_blob = cv2.erode(blob_region, erosion_kernel, iterations=1)

                # Add the eroded blob to the pre-erosion image.
                pre_erosion[labels == i] = eroded_blob

                # Remove the blob from the binary image.
                binary[labels == i] = 0

                # Add the eroded blob to the binary image.
                binary[labels == i] = eroded_blob

            # Check if the area of the blob is above the size threshold.
            elif area_blob > size_threshold:

                # Get the region of the binary image that contains the blob.
                blob_region = binary[labels == i]

                # Calculate the perimeter of the blob.
                perimeter_blob = cv2.arcLength(blob_region, True)

                # Calculate the circularity of the blob.
                circularity_blob = 4 * np.pi * (area_blob / (perimeter_blob ** 2))

                # Check if the circularity is below the threshold.
                if circularity_blob < circularity_threshold:

                    # Erode the blob in the binary image.
                    eroded_blob = cv2.erode(blob_region, erosion_kernel, iterations=1)

                    # Add the eroded blob to the pre-erosion image.
                    pre_erosion[labels == i] = eroded_blob

                    # Remove the blob from the binary image.
                    blob_region = 0

                    # Add the eroded blob to the binary image.
                    blob_region = eroded_blob

        # All blobs have been processed.
        final_image = binary

        # All blobs have been processed.
        # The unknown region must be black in color.
        binary[unknown==255] = 0

        # Update the 'markers' array with the processed blobs.
        _, markers = cv2.connectedComponents(binary)
        # Increment marker values by 1 so that the sure background area gets value as 1
        markers = markers + 1

        # Mark the unknown regions as 0 in the markers image
        markers[unknown==255] = 0

        # Merge the original image into 3 channels
        img3 = cv2.merge((red,red,red))

        # Apply the watershed algorithm
        markers = cv2.watershed(img3, markers)
        img3[markers == -1] = [255,0,0]

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers after Watershed', markers_vis_after)

        cv2.imshow('Final Result', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("done")
        self.threshImages.append(finalParticels1)
        return img3

















    def segmentImage4(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)


        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)




        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)



        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Finding elongated blobs and apply erosion to split them
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1



                # Copy the sure foreground image.
        binary = sure_fg.copy()

        # Create an empty image to store the pre-erosion blobs.
        pre_erosion = np.zeros_like(sure_fg)

        # Set a threshold for circularity.
        circularity_threshold = 0.80  # Adjust this value as needed.

        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5
        # Run connectedComponents on the sure_fg image to get the initial blobs to process.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
        blobs_to_process = list(range(1, num_labels))

        # Store the original markers image for reference.
        original_markers = markers.copy()

        # Loop until no more blobs need to be processed.
        while blobs_to_process:

            # Take a blob from the queue.
            original_blob_index = blobs_to_process.pop(0)



            # Run connectedComponents on the binary image to get the updated blobs and their stats.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Find the new index of the blob in labels
            new_blob_index = np.where(labels == original_blob_index)[0]



            if not len(new_blob_index) > 0:
                continue
            i = new_blob_index[0]
            # Only consider blobs above a certain size.
            x, y, w, h, area = stats[i]
            if area < size_threshold:
                continue
            # Calculate aspect ratio.
            aspect_ratio = float(w) / h




            blob = np.where(labels == i, 255, 0).astype('uint8')


            # Find contours of the blob.
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume there is only one contour (the blob itself).
            contour = contours[0]

            # Calculate area and perimeter (arc length).
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                perimeter = 1

                area = 0.01

            # Calculate circularity.
            circularity = 4 * np.pi * area / (perimeter ** 2)
            n = 0

            while (( circularity < circularity_threshold)  and area > size_threshold):

                print("n = ", n)


                # Erode the blob if its circularity is below the threshold.
                pre_erosion = cv2.bitwise_or(pre_erosion, blob)

                blob = cv2.erode(blob, erosion_kernel, iterations = 1)

                # Remove the original blob from the binary image and from the markers.
                binary[labels == i] = 0
                markers[labels == i] = 0
                cv2.imshow('binaryb4', binary)
                # Add the eroded blob to the binary image and to the markers.
                binary = cv2.bitwise_or(binary, blob)
                cv2.imshow('binaryafter', binary)

                # Run connectedComponents on the eroded blob to get the new blobs.
                num, labels_blob = cv2.connectedComponents(blob)




                # For each new blob...
                for j in range(1, num):
                    # Create a mask of the new blob.
                    new_blob = np.where(labels_blob == j, 255, 0).astype('uint8')

                    # Assign a unique label to the new blob in the markers image.
                    new_blob_label = markers.max() + 1
                    markers[new_blob == 255] = new_blob_label

                    # Add the new label to the queue.
                    blobs_to_process.append(new_blob_label)







                    # Break the loop if more than one blob is created after the erosion.
                if num > 2:
                    print("num greater than 2")
                    break
                n += 1

                # Find contours of the blob.
                contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Assume there is only one contour (the blob itself).
                contour = contours[0]

                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    perimeter = 1

                # Calculate circularity.
                circularity = 4 * np.pi * area / (perimeter ** 2)



        cv2.imshow('Final Erosion Result',binary)


        # Now we'll update the 'markers' array with the final blobs.
        #for i in range(1, num_labels+1):
        #    blob = np.where(labels == i, 255, 0).astype('uint8')
        #    markers[blob == 255] = i
        #    final_markers[markers == i] = 0
        #    final_markers[blob == 255] = new_blob_label
        markers = markers
        #markers[(markers != 0) & (markers != -1)] = 1
        print(markers)

        # Normalize the markers image so that the min value is 0 and max value is 255.
        markers_normalized = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the markers to the uint8 data type.
        markers_8bit = np.uint8(markers_normalized)

        # Display the markers image.
        cv2.imshow('Markers', markers_8bit)

        cv2.imshow('Sure FG', sure_fg)
        cv2.imshow('Blobs before Erosion', pre_erosion)
        #cv2.imshow('Markers', markers)
        #cv2.imshow('Original Markers', original_markers)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0


        # Visualize markers before watershed
        markers_vis_before = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers before Watershed', markers_vis_before)



        img3 = cv2.merge((red,red,red))
        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img3, markers)
        self.image1[markers == -1] = [255,0,0]

        img2[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers after Watershed', markers_vis_after)


        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        cv2.imshow('blu',img2)
        #cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)






        print("done")





    def segmentImage6(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)


        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)




        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)



        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Finding elongated blobs and apply erosion to split them
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1


        cv2.imshow('sure_fg, dist transform', sure_fg)


                # Copy the sure foreground image.
        binary = sure_fg.copy()

        cv2.imshow('first binary', binary)


        # Create an empty image to store the pre-erosion blobs.
        pre_erosion = np.zeros_like(sure_fg)


        cv2.imshow('pre erosion, should be empy', pre_erosion)

        markers_8bit = np.uint8(markers)

        # Display the markers image.
        cv2.imshow('original markers', markers_8bit)

        # Set a threshold for circularity.
        circularity_threshold = 0.80  # Adjust this value as needed.

        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5
        # Run connectedComponents on the sure_fg image to get the initial blobs to process.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
        blobs_to_process = list(range(1, num_labels))

        # Store the original markers image for reference.
        original_markers = markers.copy()

        # Loop until no more blobs need to be processed.
        while blobs_to_process:

            # Take a blob from the queue.
            original_blob_index = blobs_to_process.pop(0)



            # Run connectedComponents on the binary image to get the updated blobs and their stats.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Find the new index of the blob in labels
            new_blob_index = np.where(labels == original_blob_index)[0]



            if not len(new_blob_index) > 0:
                continue
            i = new_blob_index[0]
            # Only consider blobs above a certain size.
            x, y, w, h, area = stats[i]
            if area < size_threshold:
                continue
            # Calculate aspect ratio.
            aspect_ratio = float(w) / h




            blob = np.where(labels == i, 255, 0).astype('uint8')


            # Find contours of the blob.
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume there is only one contour (the blob itself).
            contour = contours[0]

            # Calculate area and perimeter (arc length).
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                perimeter = 1

                area = 0.01

            # Calculate circularity.
            circularity = 4 * np.pi * area / (perimeter ** 2)
            n = 0

            while (( circularity < circularity_threshold)  and area > size_threshold):

                print("n = ", n)


                # Erode the blob if its circularity is below the threshold.
                pre_erosion = cv2.bitwise_or(pre_erosion, blob)

                blob = cv2.erode(blob, erosion_kernel, iterations = 1)

                # Remove the original blob from the binary image and from the markers.
                #0 before
                binary[labels == i] = 0
                markers[labels == i] = 0
                cv2.imshow('binaryb4', binary)
                #binaryfix = cv2.bitwise_not(binary)
                #blobinv = cv2.bitwise_not(blob)
                # Add the eroded blob to the binary image and to the markers.3
                #invFix = cv2.bitwise_and(binaryfix, blobinv)
                #binary = cv2.bitwise_not(invFix)

                #binary = cv2.bitwise_or(binary, blob)
                cv2.imshow('binaryafter', binary)

                # Run connectedComponents on the eroded blob to get the new blobs.
                num, labels_blob = cv2.connectedComponents(blob)




                # For each new blob...
                for j in range(1, num):
                    # Create a mask of the new blob.
                    new_blob = np.where(labels_blob == j, 255, 0).astype('uint8')

                    # Assign a unique label to the new blob in the markers image.
                    new_blob_label = markers.max() + 1
                    markers[new_blob == 255] = new_blob_label
                    binary[new_blob == 255] = 255
                    # Add the new label to the queue.
                    blobs_to_process.append(new_blob_label)







                    # Break the loop if more than one blob is created after the erosion.
                if num > 2:
                    print("num greater than 2")
                    break
                n += 1

                # Find contours of the blob.
                contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Assume there is only one contour (the blob itself).
                contour = contours[0]

                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    perimeter = 1

                # Calculate circularity.
                circularity = 4 * np.pi * area / (perimeter ** 2)



        cv2.imshow('Final Erosion Result',binary)


        # Now we'll update the 'markers' array with the final blobs.
        #for i in range(1, num_labels+1):
        #    blob = np.where(labels == i, 255, 0).astype('uint8')
        #    markers[blob == 255] = i
        #    final_markers[markers == i] = 0
        #    final_markers[blob == 255] = new_blob_label

        # Normalize the markers image so that the min value is 0 and max value is 255.
        markers_normalized = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the markers to the uint8 data type.
        markers_8bit = np.uint8(markers_normalized)

        # Display the markers image.
        cv2.imshow('Markers', markers_8bit)

        cv2.imshow('Sure FG', sure_fg)
        cv2.imshow('Blobs before Erosion', pre_erosion)
        #cv2.imshow('Markers', markers)
        #cv2.imshow('Original Markers', original_markers)


        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0


        # Visualize markers before watershed
        markers_vis_before = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers before Watershed', markers_vis_before)



        img3 = cv2.merge((red,red,red))
        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img3, markers)
        self.image1[markers == -1] = [255,0,0]

        img2[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers after Watershed', markers_vis_after)


        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        cv2.imshow('blu',img2)
        #cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)






        print("done")





    def segmentImage7(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)


        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)




        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)



        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Finding elongated blobs and apply erosion to split them
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1


        cv2.imshow('sure_fg, dist transform', sure_fg)


                # Copy the sure foreground image.
        binary = sure_fg.copy()

        cv2.imshow('first binary', binary)


        # Create an empty image to store the pre-erosion blobs.
        pre_erosion = np.zeros_like(sure_fg)

        post_erosion = np.zeros_like(sure_fg)

        cv2.imshow('pre erosion, should be empy', pre_erosion)

        markers_8bit = np.uint8(markers)

        # Display the markers image.
        cv2.imshow('original markers', markers_8bit)

        # Set a threshold for circularity.
        circularity_threshold = 0.90  # Adjust this value as needed.

        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5




        #will be total blobs to process
        n = 1
        #loop number
        m = 0


        # Store the original markers image for reference.
        original_markers = markers.copy()



        print("test1")




        # Loop until no more blobs need to be processed.
        while not n == 0:
            print("test2")
            #pre_eroded_samples = np.zeros_like(sure_fg)
            n = 0
            #select components to run

            if m == 0:
                 # Run connectedComponents on the sure_fg image to get the initial blobs to process.
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)


                # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
                blobs_to_process = list(range(1, num_labels))



            else:
                break
            #maybe try distance trasform
            while(blobs_to_process):
                index = blobs_to_process.pop(0)
                i = index
                print("test3")




                ###############################
                #now we identify blobs

                blob = np.where(labels == i, 255, 0).astype('uint8')


                ###############################################
                #pertinent calcuations
                x, y, w, h, area = stats[i]
                aspect_ratio = max(float(w) / h,float(w) / h)

                #too small blobs get passed
                if area < size_threshold:
                    continue

                #Find contours of the blob.
                contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Assume there is only one contour (the blob itself).
                contour = contours[0]

                # Calculate area and perimeter (arc length).
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    perimeter = 1

                    area = 0.01

                circularity = 4 * np.pi * area / (perimeter ** 2)

                print(circularity, aspect_ratio, "circ + aspectratio")


                ###############################################



                #if blobs is above size threshold or below circulairty, erode

                #could make this parallel by storing the labels and eroding all at once.
                if (( circularity < circularity_threshold)  or (aspect_ratio > aspect_ratio_threshold)):
                    n += 1
                    print("erode!!")
                    #indicates blobs that will be eroded
                    pre_erosion = cv2.bitwise_or(pre_erosion, blob)


                    #cv2.imshow('pre erode binary', binary)
                    # Remove the original blob from the binary image and from the markers.
                    #0 before
                    binary[labels == i] = 0
                    markers[blob == 255] = 0
                    #markers[labels == i] = 0
                    blob = cv2.erode(blob, erosion_kernel, iterations = 1)

                    #add new eroded blob to binary, markers
                    binary = cv2.bitwise_or(binary, blob)
                    new_blob_label = markers.max() + 1
                    markers[blob == 255] = new_blob_label

                    post_erosion = cv2.bitwise_or(post_erosion, blob)

                    #cv2.imshow('pre erode', pre_erosion)
                    #binaryfix = cv2.bitwise_not(binary)
                    #blobinv = cv2.bitwise_not(blob)
                    # Add the eroded blob to the binary image and to the markers.3
                    #invFix = cv2.bitwise_and(binaryfix, blobinv)
                    #binary = cv2.bitwise_not(invFix)

                    #binary = cv2.bitwise_or(binary, blob)
                    #cv2.imshow('post erod', post_erosion)
                    #cv2.imshow('post erode binary', binary)

                    #cv2.waitKey(0)













                # Run connectedComponents on the eroded blob to get the new blobs.
                #num, labels_blob = cv2.connectedComponents(blob)




                # For each new blob...
                #for j in range(1, num):
                    # Create a mask of the new blob.
                    #new_blob = np.where(labels_blob == j, 255, 0).astype('uint8')

                    # Assign a unique label to the new blob in the markers image.
                    #new_blob_label = markers.max() + 1
                    #markers[new_blob == 255] = new_blob_label
                    #binary[new_blob == 255] = 255
                    # Add the new label to the queue.
                    #blobs_to_process.append(new_blob_label)










            m += 0




        cv2.imshow('Final Erosion Result',binary)


        # Now we'll update the 'markers' array with the final blobs.
        #for i in range(1, num_labels+1):
        #    blob = np.where(labels == i, 255, 0).astype('uint8')
        #    markers[blob == 255] = i
        #    final_markers[markers == i] = 0
        #    final_markers[blob == 255] = new_blob_label

        # Normalize the markers image so that the min value is 0 and max value is 255.
        markers_normalized = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the markers to the uint8 data type.
        markers_8bit = np.uint8(markers_normalized)

        # Display the markers image.
        cv2.imshow('Markers', markers_8bit)

        cv2.imshow('Sure FG', sure_fg)
        cv2.imshow('Blobs before Erosion', pre_erosion)

        #secondary dist transform
        dist_transform1 = cv2.distanceTransform(pre_erosion,cv2.DIST_L2,3)
        ret2, sure_fg2 = cv2.threshold(dist_transform1,0.1*dist_transform1.max(),255,0)

        cv2.imshow('distance transform 2', sure_fg2)
        #cv2.imshow('Original Markers', original_markers)


        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0


        # Visualize markers before watershed
        markers_vis_before = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers before Watershed', markers_vis_before)



        img3 = cv2.merge((red,red,red))
        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img2, markers)
        self.image1[markers == -1] = [255,0,0]

        img2[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]

        cv2.imshow('img3', img3)

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers after Watershed', markers_vis_after)


        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        cv2.imshow('blu',img2)
        #cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)






        print("done")

    def segmentImage8(self):
        red1 = self.red1


        red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)


        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)




        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        #cv2.imshow('noiseremove',opening)


        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        #cv2.imshow('sure bg',sure_bg)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        print(dist_transform)
        median_value = np.percentile(dist_transform, 50)


        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,median_value,255,0)

        #cv2.imshow('distancetransform',sure_fg)



        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        #cv2.imshow('uknown',unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Finding elongated blobs and apply erosion to split them
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        new_label = num_labels + 1


        cv2.imshow('sure_fg, dist transform', sure_fg)


                # Copy the sure foreground image.
        binary = sure_fg.copy()

        cv2.imshow('first binary', binary)


        # Create an empty image to store the pre-erosion blobs.
        pre_erosion = np.zeros_like(sure_fg)

        post_erosion = np.zeros_like(sure_fg)

        cv2.imshow('pre erosion, should be empy', pre_erosion)

        markers_8bit = np.uint8(markers)

        # Display the markers image.
        cv2.imshow('original markers', markers_8bit)

        # Set a threshold for circularity.
        circularity_threshold = 0.90  # Adjust this value as needed.

        # Set a threshold for blob size.
        size_threshold = 200  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5




        #will be total blobs to process
        n = 1
        #loop number
        m = 0


        # Store the original markers image for reference.
        original_markers = markers.copy()



        print("test1")




        # Loop until no more blobs need to be processed.
        while not n == 0:
            print("test2")
            #pre_eroded_samples = np.zeros_like(sure_fg)
            n = 0
            #select components to run

            if m == 0:
                 # Run connectedComponents on the sure_fg image to get the initial blobs to process.
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)


                # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
                blobs_to_process = list(range(1, num_labels))



            else:
                break
            #maybe try distance trasform
            while(blobs_to_process):
                index = blobs_to_process.pop(0)
                i = index
                print("test3")




                ###############################
                #now we identify blobs

                blob = np.where(labels == i, 255, 0).astype('uint8')


                ###############################################
                #pertinent calcuations
                x, y, w, h, area = stats[i]
                aspect_ratio = max(float(w) / h,float(w) / h)

                #too small blobs get passed
                if area < size_threshold:
                    continue

                #Find contours of the blob.
                contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Assume there is only one contour (the blob itself).
                contour = contours[0]

                # Calculate area and perimeter (arc length).
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    perimeter = 1

                    area = 0.01

                circularity = 4 * np.pi * area / (perimeter ** 2)

                print(circularity, aspect_ratio, "circ + aspectratio")


                ###############################################



                #if blobs is above size threshold or below circulairty, erode

                #could make this parallel by storing the labels and eroding all at once.
                if (( circularity < circularity_threshold)  or (aspect_ratio > aspect_ratio_threshold)):
                    n += 1
                    print("erode!!")
                    #indicates blobs that will be eroded
                    pre_erosion = cv2.bitwise_or(pre_erosion, blob)


                    #cv2.imshow('pre erode binary', binary)
                    # Remove the original blob from the binary image and from the markers.
                    #0 before
                    binary[labels == i] = 0
                    markers[blob == 255] = 0
                    #markers[labels == i] = 0
                    blob = cv2.erode(blob, erosion_kernel, iterations = 1)

                    #add new eroded blob to binary, markers
                    binary = cv2.bitwise_or(binary, blob)
                    new_blob_label = markers.max() + 1
                    markers[blob == 255] = new_blob_label

                    post_erosion = cv2.bitwise_or(post_erosion, blob)

                    #cv2.imshow('pre erode', pre_erosion)
                    #binaryfix = cv2.bitwise_not(binary)
                    #blobinv = cv2.bitwise_not(blob)
                    # Add the eroded blob to the binary image and to the markers.3
                    #invFix = cv2.bitwise_and(binaryfix, blobinv)
                    #binary = cv2.bitwise_not(invFix)

                    #binary = cv2.bitwise_or(binary, blob)
                    #cv2.imshow('post erod', post_erosion)
                    #cv2.imshow('post erode binary', binary)

                    #cv2.waitKey(0)













                # Run connectedComponents on the eroded blob to get the new blobs.
                #num, labels_blob = cv2.connectedComponents(blob)




                # For each new blob...
                #for j in range(1, num):
                    # Create a mask of the new blob.
                    #new_blob = np.where(labels_blob == j, 255, 0).astype('uint8')

                    # Assign a unique label to the new blob in the markers image.
                    #new_blob_label = markers.max() + 1
                    #markers[new_blob == 255] = new_blob_label
                    #binary[new_blob == 255] = 255
                    # Add the new label to the queue.
                    #blobs_to_process.append(new_blob_label)










            m += 0




        cv2.imshow('Final Erosion Result',binary)
        ret, markers = cv2.connectedComponents(binary)

        # Now we'll update the 'markers' array with the final blobs.
        #for i in range(1, num_labels+1):
        #    blob = np.where(labels == i, 255, 0).astype('uint8')
        #    markers[blob == 255] = i
        #    final_markers[markers == i] = 0
        #    final_markers[blob == 255] = new_blob_label

        # Normalize the markers image so that the min value is 0 and max value is 255.
        markers_normalized = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the markers to the uint8 data type.
        markers_8bit = np.uint8(markers_normalized)

        # Display the markers image.
        cv2.imshow('Markers', markers_8bit)

        cv2.imshow('Sure FG', sure_fg)
        cv2.imshow('Blobs before Erosion', pre_erosion)

        #secondary dist transform
        #dist_transform1 = cv2.distanceTransform(pre_erosion,cv2.DIST_L2,3)
        #ret2, sure_fg2 = cv2.threshold(dist_transform1,0.1*dist_transform1.max(),255,0)

        #cv2.imshow('distance transform 2', sure_fg2)
        #cv2.imshow('Original Markers', original_markers)


        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0


        # Visualize markers before watershed
        markers_vis_before = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers before Watershed', markers_vis_before)



        img3 = cv2.merge((red,red,red))
        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img2, markers)
        self.image1[markers == -1] = [255,0,0]

        img2[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]

        cv2.imshow('img3', img3)

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Markers after Watershed', markers_vis_after)


        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        cv2.imshow('blu',img2)
        #cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)






        print("done")

    def segmentImage9(self):
        red1 = self.red1


        #red2 = cv2.blur(red1, (3, 3))
        red = cv2.bilateralFilter(red1, d=10, sigmaColor=75, sigmaSpace=75)
        #redblur1 = cv2.bilateralFilter(red1, d=10, sigmaColor=60, sigmaSpace=75)
        #redblur2 = cv2.bilateralFilter(red1, d=10, sigmaColor=90, sigmaSpace=75)


        thresh = cv2.adaptiveThreshold(red,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                       #100,
                                      cv2.THRESH_BINARY,
                                      401,

                                      10)

        total_pixels = thresh.size
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)

        white_fraction = white_pixels / total_pixels
        self.massFract = white_fraction



        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)



        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)



        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        median_value = np.percentile(dist_transform, 75)


        #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        ret, sure_fg = cv2.threshold(dist_transform,median_value,255,0)




        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)



        # Copy the sure foreground image.
        binary = sure_fg.copy()





        #setting overall thresholding values
        # Set a threshold for blob size.
        size_threshold = 50  # Adjust this value as needed.
        aspect_ratio_threshold = 1.5
        aspect_ratio_threshold = 2.0  # Set your own aspect ratio threshold
        erosion_kernel = np.ones((3,3),np.uint8)  # Erosion kernel
        # Set a threshold for circularity.
        circularity_threshold = 0.90  # Adjust this value as needed.





        #g is number of blobs to be processed
        g = 1
        print("test1")
        images_to_combine = []
        image_to_analyze = [binary]

        #################################
        #start loop here.

        while not g == 0:
            g = 0


            currentImage = image_to_analyze[len(image_to_analyze)-1].copy()
            #current image will go to image to combine, when we remove all particles than need to be edited

            #Samples to be processed
            image_to_process = np.zeros_like(currentImage)

            #Get labels of new things to process
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats( currentImage,connectivity=8)


            # Create a queue to store blobs that need to be processed. Initially, all blobs need to be processed.
            blobs_to_process = list(range(1, num_labels))



            while(blobs_to_process):

                index = blobs_to_process.pop(0)
                i = index





                ###############################
                #now we identify blobs

                blob = np.where(labels == i, 255, 0).astype('uint8')


                ###############################################
                #pertinent calcuations
                x, y, w, h, area = stats[i]
                aspect_ratio = max(float(w) / h,float(w) / h)

                #too small blobs get passed
                if area < size_threshold:
                    continue

                #Find contours of the blob.
                contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Assume there is only one contour (the blob itself).
                contour = contours[0]

                # Calculate area and perimeter (arc length).
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    perimeter = 1

                    area = 0.01

                circularity = 4 * np.pi * area / (perimeter ** 2)

                #print(circularity, aspect_ratio, "circ + aspectratio")


                ###############################################



                #if blobs is above size threshold or below circulairty, erode

                #could make this parallel by storing the labels and eroding all at once.
                if (( circularity < circularity_threshold)  or (aspect_ratio > aspect_ratio_threshold)):

                    g = g+1

                    currentImage[labels == i] = 0



                    #add new eroded blob to binary, markers
                    image_to_process = cv2.bitwise_or(image_to_process, blob)
                    #new_blob_label = markers.max() + 1

            #eroding image
            image_to_process = cv2.erode(image_to_process, erosion_kernel, iterations = 1)

            images_to_combine.append(currentImage)
            image_to_analyze.append(image_to_process)



        #end loop here
        ####
        final_erosion = np.zeros_like(binary)
        for image in images_to_combine:
            final_erosion = cv2.bitwise_or(final_erosion, image)
            del image
        j = 0

        for image1 in image_to_analyze:
            del image1
            #cv2.imshow('Final Erosion Result'+str(j),image1)
            j = j +1


























        #cv2.imshow('Final Erosion Result',final_erosion)
        ret, markers = cv2.connectedComponents(final_erosion)

        # Now we'll update the 'markers' array with the final blobs.
        #for i in range(1, num_labels+1):
        #    blob = np.where(labels == i, 255, 0).astype('uint8')
        #    markers[blob == 255] = i
        #    final_markers[markers == i] = 0
        #    final_markers[blob == 255] = new_blob_label


        # Display the markers image.
        #cv2.imshow('Markers', markers_8bit)



        #secondary dist transform
        #dist_transform1 = cv2.distanceTransform(pre_erosion,cv2.DIST_L2,3)
        #ret2, sure_fg2 = cv2.threshold(dist_transform1,0.1*dist_transform1.max(),255,0)

        #cv2.imshow('distance transform 2', sure_fg2)
        #cv2.imshow('Original Markers', original_markers)


        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0


        # Visualize markers before watershed
        markers_vis_before = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        #cv2.imshow('Markers before Watershed', markers_vis_before)



        img3 = cv2.merge((red,red,red))
        img2 = cv2.merge((thresh,thresh,thresh))
        markers = cv2.watershed(img2, markers)
        self.image1[markers == -1] = [255,0,0]

        img2[markers == -1] = [255,0,0]
        opening[markers == -1] = [0]

        #cv2.imshow('img3', img3)

        # Visualize markers after watershed
        markers_vis_after = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
        #cv2.imshow('Markers after Watershed', markers_vis_after)


        # Find contours
        contours, _ = cv2.findContours(markers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours on
        contour_image = np.zeros_like(red1)


        finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

        finalParticels1 = finalParticels
        #finalParticels1 = cv2.medianBlur(finalParticels, 3)
        #cv2.imshow('blu',img2)
        #cv2.waitKey(0)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        self.threshImages.append(finalParticels1)






        print("done")





    def detectBlobs1(self):
        print("hello")
        finalParticels = self.threshImages[0]
        finalParticels = cv2.bitwise_not(finalParticels)
        finalParticels2 = finalParticels.copy()
        #detected_circles = cv2.HoughCircles(finalParticels,
        #                   cv2.HOUGH_GRADIENT, 1, 50, param1 = 200,
        #               param2 = 100, minRadius = 150, maxRadius = 10000)
        # Set our filtering parameters
        # Initialize parameter setting using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Set Area filtering parameters
        params.filterByArea = False
        params.minArea = 100

        # Set Circularity filtering parameters
        params.filterByCircularity = True
        params.minCircularity = 0.6

        # Set Convexity filtering parameters
        params.filterByConvexity = True
        params.minConvexity = 0.2

        # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(finalParticels)

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(finalParticels, keypoints, blank, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(keypoints)
        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(len(keypoints))
        cv2.putText(blobs, text, (12, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        node_list = []
        SumR = 0
        node_number = 0
        MaxR = 0
        for i in keypoints:
            nodething = (node_number,{"pos": (i.pt[0],i.pt[1]) , "d" : i.size})
            SumR += i.size
            node_list.append(nodething)

            if i.size>MaxR:
                MaxR = i.size


            #add text label to each circle
            #cv2.putText(blobs, str(node_number), (int(i.pt[0]), int(i.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (246,255,12), 3)
            node_number += 1

        self.blobsInfo.append(node_list)

        #print(node_list)

        # Show blobs
        #cv2.imshow("Filtering Circular Blobs Only", blobs)




        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



    def detectBlobs2(self):
        # Find contours
        finalParticels = self.threshImages[0]
        finalParticels_color = cv2.cvtColor(finalParticels, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(finalParticels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the contours and ellipses on
        result = np.zeros_like(finalParticels_color)

        for contour in contours:
            # Fit an ellipse to the contour

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)

                # Draw the contour and ellipse on the image
                cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)
                cv2.ellipse(result, ellipse, (0, 255, 255), 2)

        # Display the image
        cv2.imshow("Contours and Ellipses", result)
        cv2.waitKey(4)
        cv2.destroyAllWindows()


    def runGraph(self,imageSave):
        Graph1 = graphAnalysis(self.blobsInfo[0],self.imageLocation)
        return Graph1.startup(imageSave,self.massFract)

"""






#sees if nodes are truly adjacent, then links them in a graph
def nodeComparator(node1, node2, graph):
    if node1 == node2:
        return

    Positions = nx.get_node_attributes(graph,'pos')

    Diameters = nx.get_node_attributes(graph,'d')
    DeltaXY = (Positions[node2][0]-Positions[node1][0])**2 + (Positions[node2][1]-Positions[node1][1])**2
    Comparator = 1.25*(Diameters[node2]/2+Diameters[node1]/2)**2-DeltaXY

    if Comparator >= 0:
        graph.add_edge(node1, node2)
        #print("edge added between nodes: ", node1, ", ", node2)

    return






testloc = "I:/My Drive/Research/OtherProjects/"
testloc = testloc + "OilInwaterStuff/TestImage/MineralOilTest2.tif"
#testloc = testloc + "OilInwaterStuff/TestImage/Particels.png"

img = imread(testloc)
#red = img.copy()
#red[:,:,1] = 0
#red[:,:,2] = 0
#plt.imshow(red)

image1 = cv2.imread(testloc,1)
b,g,red1 = cv2.split(image1)
# Convert to grayscale.

# Blur using 3 * 3 kernel.
red = cv2.blur(red1, (3, 3))
#ret, thresh = cv2.threshold(red,0,255,cv2.THRESH_BINARY +
#                                            cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(red,
                              255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY,
                              401,

                              10)
cv2.imshow('segment',thresh)
cv2.imshow('OG IMage',red)




# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow('noiseremove',opening)


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=2)
cv2.imshow('sure bg',sure_bg)



# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
#ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

cv2.imshow('distancetransform',sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('uknown',unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
print(thresh)
print(image1)
img2 = cv2.merge((thresh,thresh,thresh))
markers = cv2.watershed(img2,markers)
image1[markers == -1] = [255,0,0]
opening[markers == -1] = [0]
finalParticels = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, iterations = 2)

finalParticels1 = finalParticels
#finalParticels1 = cv2.medianBlur(finalParticels, 3)

cv2.imshow('blu',image1)





cv2.waitKey()
"""
