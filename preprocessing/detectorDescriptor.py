"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def unit8Image(imageGrayInput):
    try:
        imageGrayInput_unit8 = np.uint8(imageGrayInput)
        return imageGrayInput_unit8
    except:
        print 'Make sure numpy is imported and is renamed as np. Also make sure your input is a greyscale image'

def featureDescriptorORB(roiImageFiltered, kp):
    orb = cv2.ORB()
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    try:
        kp, des = orb.compute(roiImageFiltered_unit8, kp)
        roiKeyPointImage_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, color=(0, 255, 0), flags=0)
        return kp, des, roiKeyPointImage_unit8
    except TypeError:
         print 'The input image in the function featureDescriptorORB is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDescriptorORB is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDescriptorORB is not working. You have most likely given invalid arguments.'



def featureDetectCorner(roiImageFiltered):
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    fast = cv2.FastFeatureDetector()
    try:
        kp = fast.detect(roiImageFiltered_unit8, None)
        roiKeyPointImage_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, color=(255, 0, 0))
        return kp, roiKeyPointImage_unit8
    except TypeError:
         print 'The input image in the function featureDetectCorner is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDetectCorner is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDetectCorner is not working. You have most likely given invalid arguments.'


def featureDetectDesORB(roiImageFiltered):
    orb = cv2.ORB()
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    try:
        kp, des = orb.detectAndCompute(roiImageFiltered_unit8, None)
        roiKeyPointImage_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, None, (255, 0, 0), 2)
        return kp, des, roiKeyPointImage_unit8
    except TypeError:
         print 'The input image in the function featureDetectDesORB is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDetectDesORB is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDetectDesORB is not working. You have most likely given invalid arguments.'



def featureDetectDesSTAR(roiImageFiltered):
    star = cv2.FeatureDetector_create("STAR")
    brief = cv2.DescriptorExtractor_create("BRIEF")
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    try:
        kp = star.detect(roiImageFiltered_unit8, None)
        kp, des = brief.compute(roiImageFiltered_unit8, kp)
        roiKeyPointImage_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, None, (255, 0, 0), 2)
        return kp, des, roiKeyPointImage_unit8
    except TypeError:
         print 'The input image in the function featureDetectDesSTAR is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDetectDesSTAR is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDetectDesSTAR is not working. You have most likely given invalid arguments.'



def featureDetectSTAR(roiImageFiltered):
    star = cv2.FeatureDetector_create("STAR")
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    try:
        kp = star.detect(roiImageFiltered_unit8, None)
        roiKeyPointImage_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, None, (255, 0, 0), 2)
        return kp, roiKeyPointImage_unit8
    except TypeError:
         print 'The input image in the function featureDetectSTAR is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDetectSTAR is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDetectSTAR is not working. You have most likely given invalid arguments.'


def featureDescriptorBrief(roiImageFiltered, kp):
    brief = cv2.DescriptorExtractor_create("BRIEF")
    roiImageFiltered_unit8 = unit8Image(roiImageFiltered)
    try:
        kp, des = brief.compute(roiImageFiltered_unit8, kp)
        roiImageFiltered_unit8 = cv2.drawKeypoints(roiImageFiltered_unit8, kp, color=(0, 255, 0), flags=0)
        return kp, des, roiImageFiltered_unit8
    except TypeError:
         print 'The input image in the function featureDescriptorBrief is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in featureDescriptorBrief is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function featureDescriptorBrief is not working. You have most likely given invalid arguments.'


def kpToHogBox(kp):
    xValue = kp.pt[0]
    yValue = kp.pt[1]
	# length = 16
    try:
        return xValue-8, yValue-8
    except:
        print 'No key points were found in the image'
