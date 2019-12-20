"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes
import glob

fileName = "frame0002.jpg"
inputImage=cv2.imread(fileName)

cv2.SuperpixelSEEDS.createSuperpixelSEEDS(np.size(inputImage,0), np.size(inputImage,0), 3, 10, 10, use_prior = 4, histogram_bins=20, double_step = 0)
cv2.SuperpixelSEEDS.iterate(inputImage, 80)
cv2.SuperpixelSEEDS.getLabels(labels_out)
