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


svmTestDataPath ="TrainingSet/SVMCodes/Noise/Testing/TestingData.npy"
svmTestLabelPath = "TrainingSet/SVMCodes/Noise/Testing/TestingLabels.npy"

svmTestData = np.load(svmTestDataPath)
svmTestLabels = np.load(svmTestLabelPath)

svm = cv2.SVM()
svm.load("TrainingSet/SVMCodes/Noise/svmNoise.dat")

results = svm.predict_all(svmTestData)
computeAccuracy = results==svmTestLabels
print ("Accuracy of the SVM on Testing Data is: ", np.float32(np.count_nonzero(computeAccuracy))/np.float32(np.size(computeAccuracy,0))*100)
