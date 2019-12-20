# -*- coding: utf-8 -*-
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
import time
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing


global predictionArray, centroidArray, labels
global objects
global backgroundClass
predictionArray = []
centroidArray   = []
labels = []
new_contours = []

backgroundClass = 5

objects = ['waterkettle', 'milkcarton', 'trashcan', 'Background'];

def getPrediction(cnt,boundBoxContour,codeBookCenters,svm, objects):
    prediction =    None
    centroid = None
    label = None
    [x, y, w, h] = cv2.boundingRect(cnt)
            # Get the moments of the each contour for computing the centroid of the contour
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
        centroid = (cx,cy)
        # cx,cy are the centroid of the contour
        extendBBox20 = 20
        extendBBox10 = 10
        left = 0
        right = 0
        top = 0
        bottom = 0

#
		 #Extend it by 10 pixels to avoid missing the key points on the edges
        if x-extendBBox20 > 0:
            left = x-extendBBox20
        elif x-extendBBox10 > 0:
            left = x-extendBBox10
        else:
            left = x
        if y-extendBBox20 > 0:
            top = y-extendBBox20
        elif y-extendBBox10 > 0:
            top = y-extendBBox10
        else:
            top = y
        if x+w+extendBBox20 < boundBoxContour.shape[0]:
            right = x+w+extendBBox20
        elif x+w+extendBBox10 < boundBoxContour.shape[0]:
            right = x+w+extendBBox10
        else:
            right = x+w
        if y+h+extendBBox20 < boundBoxContour.shape[1]:
            bottom = y+h+extendBBox20
        elif y+h+extendBBox10 < boundBoxContour.shape[1]:
            bottom = y+h+extendBBox10
        else:
            bottom = y+h
        roiImage = boundBoxContour[top:bottom,left:right]
            #roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
            #roiImageFiltered = cv2.equalizeHist(roiImage)
        roiImageFiltered = roiImage

			# Detect the corner key points
#                kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)

                # Use the ORB feature detector and descriptor on the contour
            #time_sift_start = time.time()
        kp, des, roiKeyPointImage = detDes.featureDetectDesSIFT(roiImageFiltered)
        #time_sift_end = time.time()
        #print 'time in sift ' + str(time_sift_end - time_sift_start )
        if np.size(kp)>0:
#            time_dist_compute_start = time.time()
            histPoints = tH.histogramContour(des,codeBookCenters)
            #time_dist_compute_end = time.time()
            #print 'time in dist compute ' + str(time_dist_compute_end - time_dist_compute_start )
            #time_randomForest_start = time.time()
            prediction = tC.classify(histPoints, svm)
            #time_randomForest_end = time.time()
            #print 'time in random forest ' + str(time_randomForest_end - time_randomForest_start )
            label = objects[np.int(prediction)]
            if prediction < backgroundClass:
                predictionArray.append(prediction)
                centroidArray.append(centroid)
                labels.append(label)

            # Get the bounding box of the contour

    #return predictionArray, centroidArray, labels
                   #print prediction
				## If the predicted class is not the background class then add the prediction to the prediction array and get its centroid

def processAndClassify(frame,codeBookCenters, svm):

    #'coffeemug', 'kettleBackground','milkcartonBackground','trashcanBackground', 'mugBackground',



	# Convert the image into a grayscale image
    grayScaleInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Apply meanshift on the RGB image
    meanShiftResult = prePro.meanShift(np.uint8(frame))
    #plt.imshow(meanShiftResult)
	# Convert the result of the Mean shifted image into Grayscale
    meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	# Apply adaptive thresholding on the resulting greyscale image
    meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
#    kernel = np.ones((5,5),np.uint8)
#    opening = cv2.dilate(meanShiftAdapResult,kernel,iterations=1)
	# Draw Contours on the Input image with results from the meanshift
	# Find the contours on the mean shifted image
    contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)

    ## Use Histogram equalabs
    #boundingBoxContour = opening
    boundBoxContour = grayScaleInput.copy()
    #boundBoxContour = cv2.equalizeHist(grayScaleInput.copy())
    jobs = []
    for cnt in contours:
        if cv2.contourArea(cnt)>500:
            new_contours.append(cnt)
#        p = multiprocessing.Process(target=getPrediction,args=(cnt,boundBoxContour,codeBookCenters,svm, objects))
#        jobs.append(p)
#        p.start()
	# For each contour
    print np.size(new_contours)
    Parallel(n_jobs = 2, verbose = 10)(delayed(getPrediction)(cnt,boundBoxContour,codeBookCenters,svm, objects) for cnt in contours)

    return predictionArray, centroidArray,labels
