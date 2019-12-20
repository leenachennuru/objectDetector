# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes
import glob

rootInputName = "TrainingSet/TrainingData/objectRecognitionDatasetKinect/"
rootOutputName = "TrainingSet/newDatasetPrep/contours/"
rootMeanName = "TrainingSet/newDatasetPrep/meanShift/"
rootAdapThreshold = "TrainingSet/newDatasetPrep/adapThreshold/"
rootContourDraw = "TrainingSet/newDatasetPrep/contourDraw/"
formatName = "*.jpg"

fileList = glob.glob(rootInputName + formatName)
print ("Current Directory Name:" + rootInputName)
count = 0
for files in fileList:
	inputImage=cv2.imread(fileList[count])
	fileName = os.path.basename(fileList[count])
	print ("Processing Image " + fileList[count])
	grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

	meanShiftResult = prePro.meanShift(inputImage)
	cv2.imwrite(rootMeanName + fileName, meanShiftResult)

	meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
	cv2.imwrite(rootAdapThreshold + fileName, meanShiftAdapResult)

	contourPlot = prePro.contourDraw(inputImage, meanShiftAdapResult)
	cv2.imwrite(rootContourDraw + fileName, contourPlot)

	print ("Preprocessing Results Written.")

	contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)
	boundBoxContour = grayScaleInput.copy()
	counter = 0
	for cnt in contours:
	    if cv2.contourArea(cnt)>20:
		    print("Processing Contour no.", str(counter))
		    [x, y, w, h] = cv2.boundingRect(cnt)
		    extendBBox = 5
		    roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
		    roiImageFiltered = roiImage  #cv2.medianBlur(roiImage, 3)
		    kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
		    kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
		    if np.size(kp)>0:
		        cv2.imwrite(rootOutputName + str(count*1000 + counter*10) + ".png", roiImage)
		        print ("Path: " + rootOutputName + str(count*100 + counter*10) + ".png")
		        print ("Found some non-zero keypoints for the countours.")
		    counter = counter + 1
	count = count + 1
