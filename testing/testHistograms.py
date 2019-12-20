"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier
import glob
#import rospkg
import preObj as prePro
import detectorDescriptor2 as detDes
import os

#codeBookLabelsPath = "/Scripts/Codebook/codeBookFinal.npy"
#codeBookCentersPath = "/Scripts/CodeBook/codeBookNoiseCenters25000.npy"
#
#rospack = rospkg.RosPack()
#path_to_package = rospack.get_path("object_recognition")
#
##codeBookLabels = np.load(path_to_package + codeBookLabelsPath)
#codeBookCenters = np.load(path_to_package + codeBookCentersPath)

def histogramContour(des,codeBookCenters):
	histPoints = classifier.minDistance(des,codeBookCenters)
	return histPoints[0]
