# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor2 as detDes
import glob
import testHistograms as tH
import testClassify as tC

framecopy = frame.copy()
grayScaleInput = cv2.cvtColor(framecopy, cv2.COLOR_BGR2GRAY)
	# Apply meanshift on the RGB image
meanShiftResult = prePro.meanShift(np.uint8(framecopy))
    #plt.imshow(meanShiftResult)
	# Convert the result of the Mean shifted image into Grayscale
meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	# Apply adaptive thresholding on the resulting greyscale image
meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
#    kernel = np.ones((5,5),np.uint8)
#    opening = cv2.dilate(meanShiftAdapResult,kernel,iterations=1)
	# Draw Contours on the Input image with results from the meanshift
	# Find the contours on the mean shifted image
contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)

    ## Use Histogram equalabs
    #boundingBoxContour = opening
boundBoxContour = grayScaleInput.copy()
