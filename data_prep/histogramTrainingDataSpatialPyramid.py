# -*- coding: utf-8 -*-
"""
@author: 4chennur
"""

import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier
import glob
import preObj as prePro
import detectorDescriptor2 as detDes
import os
import spatialPyramidConst as sp


#codeBookCentersPath = "CodeBook/withNoise.npy"
#codeBookLabelsPath = "CodeBook/codeBookNoiseLabels.npy"
codeBookCentersPath = "CodeBook/siftSubsampledCenters10000.npy"


#codeBookNoise = np.float32(np.load(codeBookPath))
#codeBookLabels = np.load(codeBookLabelsPath)
codeBookCenters = np.load(codeBookCentersPath)


rootInputName = "FinalExtensiveDataMerged/"
formatName = "*.png"

listDir = [];
listDir.append("kettle-bb/")
listDir.append("milk-bb/")
listDir.append("basket-bb/")
#listDir.append("mug-bb/")
#listDir.append("kettle-bgbb/")
#listDir.append("milk-bgbb/")
#listDir.append("basket-bgbb/")
#listDir.append("mug-bgbb/")
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
		inputImage=cv2.imread(fileList[count],0)
		fileName = os.path.basename(fileList[count])
		roiImageFiltered = inputImage
		kp, des = detDes.featureDetectDesSIFT(roiImageFiltered)

		#kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
		#kp, des,roiImageFiltered = detDes.featureDescriptorORB(inputImage,kp)
		#kp,roiImageFiltered = detDes.featureDetectCorner(inputImage)
		#kp,des,roiImageFiltered = detDes.featureDescriptorORB(inputImage,kp)
		if np.size(kp)>0:
			#print np.shape(codeBookCenters)
			#print inputImage.shape[0], inputImage.shape[1]
			histPoints  = sp.buildHistogramForEachImageAtDifferentLevels(inputImage.shape[1], inputImage.shape[0], kp, des, 1,codeBookCenters)
			#print inputImage.shape[0], inputImage.shape[1]
			histogramSVM.append(histPoints)
			labels.append([dirCount])
			HistogramComputed = HistogramComputed + 1
			print ("Histogram computed for the chosen Image. directory = " + str(dirCount) + 'image number' + str(count))
		count = count + 1
	dirCount = dirCount + 1

histogramNew  = np.float32(np.array(histogramSVM))
labelsNew  = np.float32(np.array(labels))

np.save("SVM/TrainingDataSiftSpatialPyramid0.50.7510000.npy", histogramNew)
np.save("SVM/TrainingLabelsSiftSpatialPyramid0.50.7510000.npy", labelsNew)
