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


def processAndClassify(frame):
    backgroundClass = 3
    predictionArray = []
    centroidArray   = []
    # Convert the image into a grayscale image
    grayScaleInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply meanshift on the RGB image
    meanShiftResult = prePro.meanShift(frame)
    # Convert the result of the Mean shifted image into Grayscale
    meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding on the resulting greyscale image
    meanShiftAdapResult = prePro.adapThresh(meanShiftGray)
    # Draw Contours on the Input image with results from the meanshift
    contourPlot = prePro.contourDraw(frame, meanShiftAdapResult)

    #cv2.imwrite('contourPlot.png',contourPlot)
    # Find the contours on the mean shifted image
    contours, hierarchy = prePro.contourFindFull(meanShiftAdapResult)
    # Find the contours on the mean shifted image
    boundBoxContour = grayScaleInput.copy()
    frameBoundingBox = frame.copy()
    # For each contour
    for cnt in contours:
		# If the area covered by the contour is greater than 500 pixels
        if cv2.contourArea(cnt)>20:
			# Get the bounding box of the contour
            [x, y, w, h] = cv2.boundingRect(cnt)
			# Get the moments of the each contour for computing the centroid of the contour
            moments = cv2.moments(cnt)
            if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
                cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
                centroid = (cx,cy)
			# cx,cy are the centroid of the contour
                extendBBox = 10
			# Extend it by 10 pixels to avoid missing the key points on the edges
                roiImage = boundBoxContour[y-extendBBox:y+h+extendBBox, x-extendBBox:x+w+extendBBox]
                roiImageFiltered = roiImage
			# Detect the corner key points
                kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
			# Use the ORB feature detector and descriptor on the contour
                kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
                if np.size(kp)>0:
                    histPoints = tH.histogramContour(des)
                    prediction = tC.classify(np.float32(histPoints))
				## If the predicted class is not the background class then add the prediction to the prediction array and get its centroid
                    if prediction != backgroundClass:
                        cv2.rectangle(frameBoundingBox,(x,y),(x+w,y+h),(0,255,0),2)
    return predictionArray, centr
