"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier
import glob
import preObj as prePro
import detectorDescriptor as detDes
import os


codeBookPath = "TrainingSet/CodeBook/withNoise.npy"
codeBookLabelsPath = "TrainingSet/CodeBook/Noise/codeBookNoiseLabels.npy"
codeBookCentersPath = "TrainingSet/CodeBook/Noise/codeBookNoiseCenters.npy"


codeBookNoise = np.float32(np.load(codeBookPath))
codeBookLabels = np.load(codeBookLabelsPath)
codeBookCenters = np.load(codeBookCentersPath)

rootInputName = "FinalExtensiveData/"
formatName = "*.png"

listDir = [];
listDir.append("kettle-bb/")
listDir.append("milk-bb/")
listDir.append("basket-bb/")
listDir.append("mug-bb/")
listDir.append("kettle-bgbb/")
listDir.append("milk-bgbb/")
listDir.append("basket-bgbb/")
listDir.append("mug-bgbb/")
listDir.append("background/")


histogramSVM = []
labels = []
HistogramComputed = 0
# testing the things
dirCount = 0
for direct in listDir:
	fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
	print ("Current Directory Name:" + rootInputName + listDir[dirCount])
	count = 0
	for files in fileList:
		inputImage=cv2.imread(fileList[count])
		fileName = os.path.basename(fileList[count])
		roiImageFiltered = inputImage #cv2.medianBlur(inputImage, 3)
		kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
		kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
		if np.size(kp)>0:
			histPoints = classifier.minDistance(des,codeBookCenters)
			histogramSVM.append(histPoints[0])
			labels.append([dirCount])
			HistogramComputed = HistogramComputed + 1
			print ("Histogram computer for the chosen Image.")
		count = count + 1
	dirCount = dirCount + 1

histogramNew  = np.float32(np.array(histogramSVM))
labelsNew  = np.float32(np.array(labels))

np.save("TrainingSet/SVMCodes/Noise/Testing/TestingData.npy", histogramNew)
np.save("TrainingSet/SVMCodes/Noise/Testing/TestingLabels.npy", labelsNew)
