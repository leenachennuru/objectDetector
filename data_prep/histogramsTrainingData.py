"""
@author: 4chennur, 4wahab
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


#codeBookPath = "CodeBook/withNoise.npy"
#codeBookLabelsPath = "CodeBook/codeBookNoiseLabels.npy"
codeBookCentersPath = "CodeBook/siftSubsampledCenterswithHomeLab10000.npy"


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
listDir.append("BackgroundwithHomeLab/")

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
		roiImageFiltered = inputImage #kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered)

		#kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
		kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered)
		if np.size(kp)>0:
			histPoints = classifier.minDistance(des,codeBookCenters)
			histogramSVM.append(histPoints[0])
			labels.append([dirCount])
			HistogramComputed = HistogramComputed + 1
			print ("Histogram computed for the chosen Image. directory = " + str(dirCount) + 'image number' + str(count))
		count = count + 1
	dirCount = dirCount + 1

histogramNew  = np.float32(np.array(histogramSVM))
labelsNew  = np.float32(np.array(labels))

np.save("SVM/TrainingDataSiftwithHomeLab10000.npy", histogramNew)
np.save("SVM/TrainingLabelsSiftwithHomeLab10000.npy", labelsNew)
