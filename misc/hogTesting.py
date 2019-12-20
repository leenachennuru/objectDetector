"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
import preObj as prePro
import detectorDescriptor as dd

inputImage=cv2.imread('TrainingSet/TrainingSetBelow/blueBanana_48_inpImg.png')

grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

meanShiftResult = prePro.meanShift(inputImage)
meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
meanShiftAdapResult = prePro.adapThresh(meanShiftGray)

contourPlot = prePro.contourDraw(inputImage, meanShiftAdapResult)

contours, hierarchy = prePro.contourFind(meanShiftAdapResult)
boundBoxContour = grayScaleInput.copy()
counter = 0
for cnt in contours:
    if cv2.contourArea(cnt)>500:
	    [x, y, w, h] = cv2.boundingRect(cnt)
	    extendBBox = 10
	    roiImage = grayScaleInput[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
	    roiImageFiltered = roiImage # cv2.medianBlur(roiImage, 3)
	    hist = dd.hogDescriptor(roiImageFiltered)
	    cv2.imwrite('TrainingSet/FeatureKeyPoints/hogDesc56' + str(counter) + '.png', hist)
	    print counter
	    counter = counter + 1
