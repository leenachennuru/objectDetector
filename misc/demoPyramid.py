"""
@author: 4chennur, 4wahab
"""
import cv2
#import spatialPyramid as sp
import numpy as np
import detectorDescriptor2 as detDes
from scipy.cluster.vq import *
import testHistograms as tH
import time
import glob
import spatialPyramidConst as sp



#codeBookPath = "CodeBook/withNoise.npy"
codeBookCentersPath = "CodeBook/siftSubsampledCenters10000.npy"
codeBookNoise = np.float32(np.load(codeBookCentersPath))
codeBook = codeBookNoise[0:10000, :]
#print np.shape(codeBook)
#a = sp.Vocabulary(codeBook, 10)

fileName = 'FinalExtensiveDataMerged/background/frame00374.png'
img = cv2.imread(fileName,0)
#kp,roiImageFiltered = det.featureDetectCorner(img)
#kp,des,roiImageFiltered = det.featureDescriptorORB(img,kp)
kp, des = detDes.featureDetectDesSIFT(img)
print np.shape(img)
kpx = []
kpy = []
for key in kp:
    kpx.append(int(key.pt[0]))
    kpy.append(int(key.pt[1]))
print 'kpx '+ str( np.max(kpx))
print 'kpy '+ str( np.max(kpy))
startTimeA = time.time()
histPoints  = sp.buildHistogramForEachImageAtDifferentLevels(img.shape[1], img.shape[0], kp, des, 1,codeBook)
endTimeA = time.time()
print 'Time Taken for Spatial Pyramid: ' + str(endTimeA- startTimeA)



#
#img = cv2.imread(fileName,0)
#kp,roiImageFiltered = det.featureDetectCorner(img)
#kp,des,roiImageFiltered = det.featureDescriptorORB(img,kp)
#startTimeB = time.time()
#histogram  = tH.histogramContour(des,codeBook)
#endTimeB = time.time()
#print 'Time Taken for usual Histogram: ' + str(endTimeB- startTimeB)
