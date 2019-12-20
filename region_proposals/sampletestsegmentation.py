# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import cv2
import numpy as np
import cv2
#from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor2 as detDes
import glob
import testHistograms as tH
import testClassify as tC
import testSegmentedThreshold as tS
from sklearn.externals import joblib
import time
#import HIKSVM

codeBookCenters = np.load("CodeBook/siftSubsampledCenterswithHomeLab10000.npy")
svm = joblib.load("SVM/RandomForestsiftwithHomeLab10000.pkl")

framecopy = cv2.imread('frame0000.jpg')

time_start = time.time()
predictionarray, centroidarray, labelarray = tS.processAndClassify(framecopy,codeBookCenters,svm)
time_end = time.time()
print "time taken for procssing is" + str(time_end - time_start)
print predictionarray, centroidarray, labelarray
#grayScaleInput = cv2.cvtColor(framecopy, cv2.COLOR_BGR2GRAY)
#meanShiftResult = prePro.meanShift(np.uint8(framecopy))
#meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
#meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
#contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)
#boundBoxContour = framecopy.copy()
#
#count = 0
#	# For each contour
#for cnt in contours:
#    # If the area covered by the contour is greater than 500 pixels
#    if cv2.contourArea(cnt)>500:
#        # Get the bounding box of the contour
#        [x, y, w, h] = cv2.boundingRect(cnt)
#        # Get the moments of the each contour for computing the centroid of the contour
#        moments = cv2.moments(cnt)
#        if moments['m00']!=0:
#            cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
#            cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
#            centroid = (cx,cy)
#            # cx,cy are the centroid of the contour
#            extendBBox20 = 20
#            extendBBox10 = 10
#            left = 0
#            right = 0
#            top = 0
#            bottom = 0
#
##
#		 #Extend it by 10 pixels to avoid missing the key points on the edges
#            if x-extendBBox20 > 0:
#                left = x-extendBBox20
#            elif x-extendBBox10 > 0:
#                left = x-extendBBox10
#            else:
#                left = x
#            if y-extendBBox20 > 0:
#                top = y-extendBBox20
#            elif y-extendBBox10 > 0:
#                top = y-extendBBox10
#            else:
#                top = y
#            if x+w+extendBBox20 < boundBoxContour.shape[0]:
#                right = x+w+extendBBox20
#            elif x+w+extendBBox10 < boundBoxContour.shape[0]:
#                right = x+w+extendBBox10
#            else:
#                right = x+w
#            if y+h+extendBBox20 < boundBoxContour.shape[1]:
#                bottom = y+h+extendBBox20
#            elif y+h+extendBBox10 < boundBoxContour.shape[1]:
#                bottom = y+h+extendBBox10
#            else:
#                bottom = y+h
#            roiImage = boundBoxContour[top:bottom,left:right]
#            #roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
#            #roiImageFiltered = cv2.equalizeHist(roiImage)
#            roiImageFiltered = roiImage
#            count +=1
#
#			# Detect the corner key points
#            #kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
#
#            # Use the ORB feature detector and descriptor on the contour
#            kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered)
#            cv2.imwrite(str(count)+'.png',roiKeyPointImage)
#
