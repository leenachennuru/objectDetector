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
import os
from skimage.feature import ORB

rootInputName = "MoreForegroundData/"
rootOutputName = "MoreForegroundData/SkimageTesting/"
formatName = "*.jpg"
listDir = [];
#listDir.append('kettle/')
listDir.append("basket1/")
#listDir.append("basket2/")
#listDir.append("basket3/")
#listDir.append("basket4/")
#listDir.append("milk/")
#listDir.append("TrainingSetOld/apple/green/")
#listDir.append("TrainingSetOld/apple/red/")
#listDir.append("TrainingSetOld/banana/green/")
#listDir.append("TrainingSetOld/banana/yellow/")
#listDir.append("TrainingSetOld/cube/blue/")
#listDir.append("TrainingSetOld/cube/yellow/")
#listDir.append("TrainingSetOld/phone/green/")
#listDir.append("TrainingSetOld/phone/yellow/")
#listDir.append("TrainingSetNao/banana/red/")
#listDir.append("TrainingSetNao/banana/yellow/")
#listDir.append("TrainingSetNao/cube/yellow/")
#listDir.append("TrainingSetNao/cube/red/")
#listDir.append("TrainingSetNao/phone/green/")
#listDir.append("TrainingSetNao/phone/red_blinds_closed/")
#listDir.append("TrainingSetNao/phone/red_blinds_open/")
#listDir.append("TrainingSetNew/apple/")
#listDir.append("TrainingSetNew/banana/")
#listDir.append("TrainingSetNew/cube/")
#listDir.append("TrainingSetOccMul/")
dirCount = 0
for direct in listDir:
    fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
    print ("Current Directory Name:" + rootInputName + listDir[dirCount])
    count = 0
    for files in fileList:
        frame=cv2.imread(fileList[count])
        base = os.path.basename(files)
        fileName = os.path.splitext(base)[0]
        print ("Processing Image " + fileList[count])
        grayScaleInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Apply meanshift on the RGB image
        meanShiftResult = prePro.meanShift(frame)
	# Convert the result of the Mean shifted image into Grayscale
        meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	# Apply adaptive thresholding on the resulting greyscale image

        meanShiftAdapResult = prePro.adapThresh(meanShiftGray)

        '''########################################################################################################################

                            Uncomment the following lines to add morphology
                            Comment the following lines if morphology need not be used

    ####################################### Morphology begins ##########################################################################'''
#        kernel = np.ones((5,5),np.uint8)
#        opening = cv2.dilate(meanShiftAdapResult,kernel,iterations=1)
#	# Draw Contours on the Input image with results from the meanshift
#        contourPlot = prePro.contourDraw(frame, opening)
#        cv2.imwrite('contourplot.png',contourPlot)
#	# Find the contours on the mean shifted image
#        contours, hierarchy = prePro.contourFindFull(opening)
#	# Draw Contours on the Input image with results from the meanshift


        '''##############################################Morphology ends ##########################################################################

                            Comment the following lines to add morphology
                            Uncomment them if no morphology needs to be used

    ####################################   No Morphology begins ########################################################################'''

        contourPlot = prePro.contourDraw(frame, meanShiftAdapResult)
    #cv2.imshow('contourplot',contourPlot)
	# Find the contours on the mean shifted image
        contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)

        '''########################################  No Morphology ends ########################################################################'''
    ## Use Histogram equalization

        boundBoxContour = grayScaleInput.copy()
        counter = 0
        for cnt in contours:
            if cv2.contourArea(cnt)>500:
                print("Processing Contour no.", str(counter))
                [x, y, w, h] = cv2.boundingRect(cnt)
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
                roiImageFiltered = roiImage  #cv2.medianBlur(roiImage, 3)
                #kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
                featureextract = ORB()
                kp, des = featureextract.detect_and_extract(roiImageFiltered)
                if (np.size(kp)>0):
                    roiKeyPointImage = cv2.drawKeypoints(roiImageFiltered, kp, color=(255, 0, 0))
                if np.size(kp)>0:
#                    print 'directory number ' + str(dirCount*10000)
#                    print 'file number ' + str(count*1000)
#                    print 'contour number ' + str(counter)
#                    print 'resulting image number ' +  str(dirCount*10000+ count*1000 + counter)
                    cv2.imwrite(rootOutputName+listDir[dirCount]+fileName + str(counter) + ".png", roiImage)
                    cv2.imwrite(rootOutputName+listDir[dirCount]+fileName + str(counter) + "kp" + ".png", roiKeyPointImage)
                    print ("Path: " +rootOutputName + str(dirCount*1000+ count*100 + counter*10) + ".png")
                    print ("Found some non-zero keypoints for the countours.")
                counter = counter + 1
        count = count + 1
    dirCount = dirCount + 1
