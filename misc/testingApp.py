"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes


# inputImage=cv2.imread('TrainingSet/TrainingSetBelow/BlueApple_55_inpImg.png')
otsuBinarization = 'otsuBinarization/'
otsuContours = 'otsuContours/'
meanShiftOtsu = 'meanShiftOtsu/'
meanShiftOtsuContours = 'meanShiftOtsuContours/'
meanShiftAda = 'meanShiftAda/'
meanShiftAdaContours = 'meanShiftAdaContours/'
cannyEdges = 'cannyEdges/'
cannyEdgesContours = 'cannyEdgesContours/'
meanShiftCanny = 'meanShiftCanny/'
meanShiftCannyContours = 'meanShiftCannyContours/'

inputFileName = 'yellowdice.png'
inputImage = cv2.imread(inputFileName)
grayScaleInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
#
#
# Otsu Result
otsuResult = prePro.otsuBin(grayScaleInput)
otsuContourPlot = prePro.contourDraw(grayScaleInput, otsuResult)


# meanShiftOtsu Result
meanShiftResult = prePro.meanShift(inputImage)
meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
meanShiftOtsuResult = prePro.otsuBin(meanShiftGray)
mOtsuContourPlot = prePro.contourDraw(grayScaleInput, meanShiftOtsuResult)

# Adaptive Threshold
meanShiftAdaResult = prePro.adapThresh(meanShiftGray)
meanShiftAdaContourPlot =  prePro.contourDraw(grayScaleInput, meanShiftAdaResult)

# Canny Edges Result
cannyEdgesResult = prePro.cannyEdge(grayScaleInput)
cannyEdgesContourPlot = prePro.contourDraw(grayScaleInput, cannyEdgesResult)

# MeanShift Canny
meanCannyResult = prePro.cannyEdge(meanShiftGray)
meanCannyContourPlot = prePro.contourDraw(grayScaleInput, meanCannyResult)

# Writing Results
cv2.imwrite(otsuBinarization + inputFileName, otsuResult)
cv2.imwrite(otsuContours + inputFileName, otsuContourPlot)

cv2.imwrite(meanShiftOtsuContours+inputFileName, mOtsuContourPlot)
cv2.imwrite(meanShiftOtsu+inputFileName, meanShiftOtsuResult)

cv2.imwrite(meanShiftAda+inputFileName, meanShiftAdaResult)
cv2.imwrite(meanShiftAdaContours+inputFileName, meanShiftAdaContourPlot)

cv2.imwrite(cannyEdges+inputFileName, cannyEdgesResult)
cv2.imwrite(cannyEdgesContours+inputFileName, cannyEdgesContourPlot)

cv2.imwrite(meanShiftCanny+inputFileName, meanCannyResult)
cv2.imwrite(meanShiftCannyContours+inputFileName, meanCannyContourPlot)

#
