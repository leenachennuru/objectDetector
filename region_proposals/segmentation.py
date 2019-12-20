"""
@author: 4chennur, 4wahab
"""
# Segmentation Approaches for Object Recognition
import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv
import Image
from scipy.ndimage import label
##import pymeanshift as pms
from time import time
from cv import *


def otsuBin(imageGrayInput):
    ret, thresh = cv2.threshold(imageGrayInput, 0, 255,
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


def meanShift(imageInput):
    meanShifted = cv2.pyrMeanShiftFiltering(imageInput, 50, 50)
    return meanShifted


def cannyEdge(imageInput):
    grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    edgeDetection = cv2.Canny(grayScaleInput, 50, 50)
    return edgeDetection


def adapThresh(imageInput):
    grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    threshAdaptive = cv2.adaptiveThreshold(grayScaleInput, 255, 1, 1, 11, 2)
    return threshAdaptive


def contourFind(prepImage):
    contours, hierarchy = cv2.findContours(prepImage, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def featureDetectDesORB(roiImageFiltered):
    orb = cv2.ORB()
    kp, des = orb.detectAndCompute(roiImageFiltered, None)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, None, (255, 0, 0), 2)
    return kp, des, roiKeyPointImage


def featureDescriptorORB(roiImageFiltered, kp):
    orb = cv2.ORB()
    kp, des = orb.compute(roiImageFiltered, kp)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(0, 255, 0), flags=0)
    return kp, des, roiKeyPointImage


def featureDetectCorner(roiImageFiltered):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(roiImageFiltered, None)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
    return kp, roiKeyPointImage


# to be ignored function
def featureDetectDesSIFT(roiImageFiltered):
    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
    kp = detector.detect(roiImageFiltered)
    kp, des = descriptor.compute(roiImageFiltered, kp)


def featureDetectDesCorner(roiImageFiltered):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(roiImageFiltered, None)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
    return kp, roiKeyPointImage

inputImage=cv2.imread('TrainingSet/TrainingSetBelow/BlueApple_55_inpImg.png')
# inputImage = cv2.imread('Lenna.png')
grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Otsu Binarization
outputOtsu = otsuBin(grayScaleInput)

# Mean Shift Prefiltering
outputMeanShift = meanShift(inputImage)

# Mean Shift Prefiltering and Otsu
grayMeanShift = cv2.cvtColor(outputMeanShift, cv2.COLOR_BGR2GRAY)
outputMeanShiftOtsu = otsuBin(grayMeanShift)

# Canny Edge Detection
cannyEdges = cannyEdge(inputImage)

# adaptive Thresholding for Image
threshAdaptive = adapThresh(inputImage)

# Finding Contours on a preprocessed Image
contours, hierarchy = contourFind(outputMeanShiftOtsu)
inputImageCopy = inputImage.copy()
contourImage = cv2.drawContours(inputImage, contours, -1, (0, 255, 0), -1)

boundBoxContour = grayScaleInput.copy()
counter = 0
for cnt in contours:
    # Bounding Box around the contour
    [x, y, w, h] = cv2.boundingRect(cnt)
    cv2.rectangle(boundBoxContour, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # the bounding box is the region of interest representative of the object
    extendBBox = 20
    roiImage = grayScaleInput[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]

    # Median Filtering on the object proposal or ROI
    roiImageFiltered = cv2.medianBlur(roiImage, 3)

    # Extract ORB Features from the object
    kp, drawnKeyPoints = featureDetectCorner(roiImageFiltered)
    cv2.imwrite('kp' + str(counter) + '.png', drawnKeyPoints)
    counter = counter + 1

# kp, drawnKeyPoints = featureDetectCorner(grayScaleInput)
# Image Saving and Output Saving
cv2.imwrite('otsuOutput.png', outputOtsu)
cv2.imwrite('meanShift.png', outputMeanShift)
cv2.imwrite('meanShiftOtsu.png', outputMeanShiftOtsu)
cv2.imwrite('cannyOutput.png', cannyEdges)
cv2.imwrite('adaptiveThres.png', threshAdaptive)

# Image Display and Output Display
while 1:
    cv2.imshow('otsu', outputOtsu)
    cv2.imshow('meanShift', outputMeanShift)
    cv2.imshow('meanShiftOtsu', outputMeanShiftOtsu)
    cv2.imshow('cannyEdgeDetection', cannyEdges)
    cv2.imshow('adaptiveThresh', threshAdaptive)
    cv2.imshow('contourImage', boundBoxContour)
    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
