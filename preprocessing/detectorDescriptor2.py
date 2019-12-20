"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
#from matplotlib import pyplot as plt


def featureDescriptorORB(roiImageFiltered, kp):
    orb = cv2.ORB()
    kp, des = orb.compute(roiImageFiltered, kp)
    if (np.size(kp)>0):
        roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, des, roiKeyPointImage
    else:
        return kp, des, roiImageFiltered


def featureDetectCorner(roiImageFiltered):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(roiImageFiltered, None)
    if (np.size(kp)>0):
        roiKeyPointImage = roiImageFiltered #cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, roiKeyPointImage
    else:
        return kp, roiImageFiltered


def featureDetectDesORB(roiImageFiltered):
    orb = cv2.ORB()
    kp, des = orb.detectAndCompute(roiImageFiltered, None)
    if (np.size(kp)>0):
        roiKeyPointImage = roiImageFiltered#cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, des, roiKeyPointImage
    else:
        return kp, des, roiImageFiltered

def featureDetectDesSIFT(roiImageFiltered):
    siftDetector = cv2.FeatureDetector_create("SIFT")
    siftDescriptor = cv2.DescriptorExtractor_create("SIFT")
    kp = siftDetector.detect(roiImageFiltered, None)
    kp, des = siftDescriptor.compute(roiImageFiltered, kp)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
#    if (np.size(kp)>0):
#        roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0)) #roiImageFiltered
#        return kp, des, roiKeyPointImage
    #else:
    return kp, des, roiKeyPointImage

def featureDetectDesSTAR(roiImageFiltered):
    star = cv2.FeatureDetector_create("STAR")
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp = star.detect(roiImageFiltered, None)
    kp, des = brief.compute(roiImageFiltered, kp)
    if (np.size(kp)>0):
        roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, des, roiKeyPointImage
    else:
        return kp, des, roiImageFiltered


def featureDetectSTAR(roiImageFiltered):
    star = cv2.FeatureDetector_create("STAR")
    kp = star.detect(roiImageFiltered, None)
    if (np.size(kp)>0):
        roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, roiKeyPointImage
    else:
        return kp, roiImageFiltered


def featureDescriptorBrief(roiImageFiltered, kp):
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp, des = brief.compute(roiImageFiltered, kp)
    if (np.size(kp)>0):
        roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
        return kp, des, roiKeyPointImage
    else:
        return kp, des, roiImageFiltered


def kpToHogBox(kp):
	xValue = kp.pt[0]
	yValue = kp.pt[1]
	# length = 16
	return xValue-8, yValue-8
