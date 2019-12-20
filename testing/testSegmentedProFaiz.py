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


def processAndClassify(frame):
	backgroundClass = 3
	predictionArray = []
	centroidArray   = []
	grayScaleInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	meanShiftAdapResult = prePro.adapThresh(grayScaleInput)
	contours, hierarchy = prePro.contourFind(meanShiftAdapResult)
	for cnt in contours:
		if cv2.contourArea(cnt)>500:
			[x, y, w, h] = cv2.boundingRect(cnt)
			extendBBox = 5
	           roiImage = grayScaleInput[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
			kp, des, roiImageKeyPoints = detDes.featureDetectDesORB(roiImage)
			if np.size(kp)>0:

	return roiImageKeyPoints
