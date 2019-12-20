# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

#!/usr/bin/env python

## Importing all the relevant modules
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor2 as detDes
import glob
import os

## Set the path to the input folder and the output folder
rootInputName = "FinalExtensiveDataMerged/"

##Format in which the files should be created
formatName = "*.png"

## List of directories from which the data needs to be collected

listDir = [];
listDir.append("kettle-bb/")
listDir.append("milk-bb/")
listDir.append("basket-bb/")
#listDir.append("mug-bb/")
listDir.append("BackgroundwithHomeLab/")


##Initializing the codebook and the number of directories that have been explored so far to collect the data
codeBook = np.zeros((1, 128))

dirCount = 0
for direct in listDir:
	fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
	print ("Current Directory Name:" + rootInputName + listDir[dirCount])
	count = 0
	for files in fileList:
		inputImage=cv2.imread(fileList[count])
		fileName = os.path.basename(fileList[count])
		roiImageFiltered = inputImage
		kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered)
		if np.size(kp)>0:
			codeBook = np.concatenate((codeBook, des), axis =0)
			print ("Found some non-zero keypoints for the countours.")
		count = count + 1
	dirCount = dirCount + 1
codeBook = codeBook[1:np.size(codeBook,0), :]
np.save("CodeBook/SiftSubsampledBackgroundwithHomeLab.npy", codeBook)
