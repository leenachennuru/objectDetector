"""
@author: 4chennur, 4wahab
"""
# Segmentation Approaches for Object Recognition
import numpy as np
import cv2
import Image
from matplotlib import pyplot as plt

def otsuBin(imageGrayInput):
    ret, thresh = cv2.threshold(imageGrayInput,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


inputImage = cv2.imread('TrainingSet/TrainingSetBelow/BlueApple_55_inpImg.png')
grayScaleInput = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)

output = otsuBin(grayScaleInput)

# Image Saving and Output Saving
cv2.imwrite('gray.png', grayScaleInput)
cv2.imwrite('otsuOutput.png', output)
cv2.imshow('otsu', output)
WaitKey ()
