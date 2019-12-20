# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


# KeyPoints Understanding
def featureDetectORB(roiImageFiltered):
    orb = cv2.ORB()
    kp, des = orb.detectAndCompute(roiImageFiltered, None)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, None, (255, 0, 0), 2)
    return kp, des, roiKeyPointImage


def featureDetectSUFT(roiImageFiltered):
    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
    kp = detector.detect(roiImageFiltered)
    kp, des = descriptor.compute(roiImageFiltered, kp)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, None, (255, 0, 0), 2)
    return roiKeyPointImage


def featureDetectCorner(roiImageFiltered):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(roiImageFiltered, None)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
    return kp, roiKeyPointImage


def featureDescriptorORB(roiImageFiltered, kp):
    orb = cv2.ORB()
    kp, des = orb.compute(roiImageFiltered, kp)
    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(0, 255, 0), flags=0)
    return kp, des, roiKeyPointImage


inputImage=cv2.imread('Lenna.png')
grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
kp, roiKeyPointImageCorner = featureDetectCorner(grayScaleInput)
kp, des, roiKeyPointImageORB = featureDescriptorORB(grayScaleInput, kp)
cv2.imwrite('CornerKeyPointsLenna.png', roiKeyPointImageCorner)
cv2.imwrite('ORBKeyPointsLenna.png', roiKeyPointImageORB)
