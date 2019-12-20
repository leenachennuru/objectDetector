"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier

codeBookPath = "CodeBook/SiftSubsampledBackgroundwithHomeLab.npy"

#KMeans code from the classifier
codeBookNoise = np.float32(np.load(codeBookPath))
print ("Creating codebook from the noise data")

compactNoise, labelsNoise, centersNoise = classifier.kmeansCodeBook(codeBookNoise, 10000)
print ("Creating codebook from the noise data")

np.save("CodeBook/siftSubsampledLabelswithHomeLab10000.npy", labelsNoise)
np.save("CodeBook/siftSubsampledCenterswithHomeLab10000.npy", centersNoise)

print ("kmneans data saved")
