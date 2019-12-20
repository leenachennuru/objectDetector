# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:19:13 2016

@author: 4chennur, 4wahab
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

import glob


#### Convnet Imports ####
from Utils import DataUtil
from Networks import MCCNN
import MCCNNExperiments
import cv2
import os
import glob


#### Preprocessing ####
import preObj as prePro

def processAndClassify(frame,networkTopology, network,trainingParameters):
    objects = ['basket', 'kettle', 'milk', 'mug', 'Background'];
    #'coffeemug', 'kettleBackground','milkcartonBackground','trashcanBackground', 'mugBackground',
    backgroundClass = 3
    predictionArray = []
    centroidArray   = []
    labels = []
	# Convert the image into a grayscale image
    framecopy = frame.copy()
    #grayScaleInput = cv2.cvtColor(framecopy, cv2.COLOR_BGR2GRAY)
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
    boundBoxContour = framecopy.copy()
    #boundBoxContour = cv2.equalizeHist(grayScaleInput.copy())
    #heatmap = frame.copy()
    #heatmapFromZeros = np.zeros(np.shape(frame))
    count = 0
	# For each contour
    for cnt in contours:
        prediction = None
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
                extendBBox20 = 20
                extendBBox10 = 10
                left = 0
                right = 0
                top = 0
                bottom = 0

#
		 #Extend it by 10 pixels to avoid missing the key points on the edges
                if x-extendBBox20 > 0:
                    left = x-extendBBox20
                elif x-extendBBox10 > 0:
                    left = x-extendBBox10
                else:
                    left = x
                if y-extendBBox20 > 0:
                    top = y-extendBBox20
                elif y-extendBBox10 > 0:
                    top = y-extendBBox10
                else:
                    top = y
                if x+w+extendBBox20 < boundBoxContour.shape[0]:
                    right = x+w+extendBBox20
                elif x+w+extendBBox10 < boundBoxContour.shape[0]:
                    right = x+w+extendBBox10
                else:
                    right = x+w
                if y+h+extendBBox20 < boundBoxContour.shape[1]:
                    bottom = y+h+extendBBox20
                elif y+h+extendBBox10 < boundBoxContour.shape[1]:
                    bottom = y+h+extendBBox10
                else:
                    bottom = y+h
                roiImage = boundBoxContour[top:bottom,left:right]
                #roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
                #roiImageFiltered = cv2.equalizeHist(roiImage)
                roiImageFiltered = cv2.resize(roiImage,(100,100))

                count +=1

                roiImageFiltered,roiImageFilteredframe = DataUtil.prepareDataLive(roiImageFiltered, DataUtil.DATA_MODALITY["Image"], networkTopology[6][0][0][4])
                prediction = MCCNN.classify(network[len(network)-1],[roiImageFiltered],trainingParameters[4])[0]



                if prediction < backgroundClass and prediction != 1:
                    predictionArray.append(prediction)
                    centroidArray.append(centroid)
                    labels.append(objects[np.int(prediction)])

                #heatmap[y:y+h,x:x+w] = prediction
                #heatmapFromZeros[y:y+h,x:x+w] = prediction
                ####################  Code using SIFT/ORB Features  ###########################
			# Detect the corner key points
#                kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
#
#                # Use the ORB feature detector and descriptor on the contour
#                kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered, kp)
#                if np.size(kp)>0:
#                   histPoints = tH.histogramContour(des,codeBookCenters)
#                   prediction = tC.classify(histPoints, svm)


                   #print prediction
				## If the predicted class is not the background class then add the prediction to the prediction array and get its centroid
        #heatmapFromZeros.convertTo(heatmap, cv2.CV_8UC1)
        #plt.imshow(heatmapFromZeros)
    return predictionArray, centroidArray,labels
