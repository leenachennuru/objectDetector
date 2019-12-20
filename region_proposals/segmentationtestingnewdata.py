# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor2 as detDes
import glob
import os

formatName = '*.jpg'
fileList = glob.glob('dataset/' + formatName)

for fileName in fileList:
    image = cv2.imread(fileName)

    grayScaleInput = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply meanshift on the RGB image
    meanShiftResult = prePro.meanShift(image)
	# Convert the result of the Mean shifted image into Grayscale
    meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	# Apply adaptive thresholding on the resulting greyscale image
    meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
	# Draw Contours on the Input image with results from the meanshift
    contourPlot = prePro.contourDraw(image, meanShiftAdapResult)
    cv2.imshow('contourplot',contourPlot)
    	# Find the contours on the mean shifted image
    contours, hierarchy = prePro.contourFind(meanShiftAdapResult)

 ## Use Histogram equalization
    boundBoxContour = grayScaleInput.copy()
    count = 0
	# For each contour
    for cnt in contours:
        # If the area covered by the contour is greater than 500 pixels
        if cv2.contourArea(cnt)>500:
            # Get the bounding box of the contour
            [x, y, w, h] = cv2.boundingRect(cnt)
            # Get the moments of the each contour for computing the centroid of the contour
            moments = cv2.moments(cnt)
            if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
                cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
                centroid = (cx,cy)
                # cx,cy are the centroid of the contour
                extendBBox = 20
		# Extend it by 10 pixels to avoid missing the key points on the edges

                roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
                count +=1
                nameImage = os.path.basename(fileName)
                cv2.imwrite('dataset/contour' + str(count) + nameImage,roiImage)

                roiImageFiltered = roiImage
                kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)

                kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)

                cv2.imwrite('dataset/CornerOrb' + str(count) + nameImage,roiKeyPointImage)
                print 'keypointimages written' +  str(count)
