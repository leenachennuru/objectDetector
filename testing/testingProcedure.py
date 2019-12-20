"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes
import glob

rootInputName = "TrainingSet/TrainingData/"
rootOutputName = "TrainingSet/ContoursFullFilter/"
formatName = "*.png"
listDir = [];
listDir.append("TrainingSetOld/apple/green/")
listDir.append("TrainingSetOld/apple/red/")
listDir.append("TrainingSetOld/banana/green/")
listDir.append("TrainingSetOld/banana/yellow/")
listDir.append("TrainingSetOld/cube/blue/")
listDir.append("TrainingSetOld/cube/yellow/")
listDir.append("TrainingSetOld/phone/green/")
listDir.append("TrainingSetOld/phone/yellow/")
listDir.append("TrainingSetNao/banana/red/")
listDir.append("TrainingSetNao/banana/yellow/")
listDir.append("TrainingSetNao/cube/yellow/")
listDir.append("TrainingSetNao/cube/red/")
listDir.append("TrainingSetNao/phone/green/")
listDir.append("TrainingSetNao/phone/red_blinds_closed/")
listDir.append("TrainingSetNao/phone/red_blinds_open/")
listDir.append("TrainingSetNew/apple/")
listDir.append("TrainingSetNew/banana/")
listDir.append("TrainingSetNew/cube/")
dirCount = 0
for direct in listDir:
	fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
	print ("Current Directory Name:" + rootInputName + listDir[dirCount])
	count = 0
	for files in fileList:
		inputImage=cv2.imread(fileList[count])
		print ("Processing Image " + fileList[count])
		grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

		meanShiftResult = prePro.meanShift(inputImage)
		meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
		meanShiftAdapResult = prePro.adapThresh(meanShiftGray)

		contourPlot = prePro.contourDraw(inputImage, meanShiftAdapResult)

		contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)
		boundBoxContour = grayScaleInput.copy()
		counter = 0
		for cnt in contours:
		    if cv2.contourArea(cnt)>500:
			    print("Processing Contour no.", str(counter))
			    [x, y, w, h] = cv2.boundingRect(cnt)
			    extendBBox = 10
			    roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
			    roiImageFiltered = cv2.medianBlur(roiImage, 3)
			    kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
    			    kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
			    if np.size(kp)>0:
			        cv2.imwrite(rootOutputName + listDir[dirCount] + str(counter) + ".png", roiKeyPointImage)
			        print ("Path: " +rootOutputName + listDir[dirCount] + str(counter) + ".png")
			        print ("Found some non-zero keypoints for the countours.")
			    counter = counter + 1
		count = count + 1
	dirCount = dirCount + 1
