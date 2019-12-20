"""
@author: 4chennur, 4wahab
"""
import cv2
#import spatialPyramid as sp
import numpy as np
import detectorDescriptor2 as det
from scipy.cluster.vq import *
import testHistograms as tH
import time
import glob

def buildHistogramForEachImageAtDifferentLevels(width, height, kp, descriptors, level,vocabulary):
        widthStep = int(width / 4)
        heightStep = int(height / 4)
        descriptors = descriptors
        # level 2, a list with size = 16 to store histograms at different location
        histogramOfLevelTwo = np.zeros((16, np.size(vocabulary, 0)))
        #print np.shape(histogramOfLevelTwo)
        for i in range(0,np.size(kp)):
            x = int(kp[i].pt[0])
            y = int(kp[i].pt[1])

            boundaryIndex = int(x / widthStep)  + int(y / heightStep) *4
            if boundaryIndex > 15:
                boundaryIndex = 15

            #print boundaryIndex
            feature = descriptors[i]
            shape = feature.shape[0]
            feature = feature.reshape(1, shape)

            codes, distance = vq(feature, vocabulary)
            #print 'codes' + str(codes[0])
            histogramOfLevelTwo[boundaryIndex][codes[0]] += 1
        # level 1, based on histograms generated on level two
        histogramOfLevelOne = np.zeros((4, np.size(vocabulary,0)))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]
        # level 0
        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]

        if level == 0:
            return histogramOfLevelZero

        elif level == 1:
            tempZero = histogramOfLevelZero.flatten() * 0.5
            tempOne = histogramOfLevelOne.flatten() * 0.75
            result = np.concatenate((tempZero, tempOne))
            return result

        elif level == 2:

            tempZero = histogramOfLevelZero.flatten() * 0.25
            tempOne = histogramOfLevelOne.flatten() * 0.25
            tempTwo = histogramOfLevelTwo.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne, tempTwo))
            return result

        else:
            return None
