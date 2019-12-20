"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
#from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor2 as detDes
import glob
import testHistograms as tH
import testClassify as tC
import time


def processAndClassify(frame,codeBookCenters, svm):
    objects = ['waterkettle', 'milkcarton', 'trashcan', 'Background'];
    #'coffeemug', 'kettleBackground','milkcartonBackground','trashcanBackground', 'mugBackground',
    backgroundClass = 3
    predictionArray = []
    centroidArray   = []
    contourArray = []
    labels = []
	# Convert the image into a grayscale image

    grayScaleInput = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Apply meanshift on the RGB image
    #meanShiftResult = prePro.meanShift(np.uint8(frame))
    meanShiftResult = cv2.GaussianBlur(np.uint8(grayScaleInput),(11,11),0)
    #plt.imshow(meanShiftResult)
	# Convert the result of the Mean shifted image into Grayscale
    #meanShiftGray = cv2.cvtColor(meanShiftResult, cv2.COLOR_BGR2GRAY)
	# Apply adaptive thresholding on the resulting greyscale image
    meanShiftAdapResult = prePro.adapThresh(meanShiftResult)
#    kernel = np.ones((5,5),np.uint8)
#    opening = cv2.dilate(meanShiftAdapResult,kernel,iterations=1)
	# Draw Contours on the Input image with results from the meanshift
	# Find the contours on the mean shifted image
    contours, hierarchy = prePro.contourFind(meanShiftAdapResult)

    ## Use Histogram equalabs
    #boundingBoxContour = opening
    boundBoxContour = grayScaleInput.copy()
    #boundBoxContour = cv2.equalizeHist(grayScaleInput.copy())
    timeComp = time.time()
    cv2.imwrite(str(timeComp) + 'Image.png',boundBoxContour)
    count = 0
	# For each contour
    for cnt in contours:
        #time_each_contour_start = time.clock()
        # If the area covered by the contour is greater than 500 pixels
        if cv2.contourArea(cnt)>500:
 #           if ...
            # Get the bounding box of the contour
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
                #roiImageFiltered = roiImage
                count +=1

			# Detect the corner key points
#                kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
                # Use the ORB feature detector and descriptor on the contour
#                time_sift_start = time.time()
                kp, des, roiKey = detDes.featureDetectDesSIFT(roiImage)
                #cv2.imwrite(str(timeComp) +'Contour' + str(count) + 'KeyPoint.png',roiKey)
#                time_sift_end = time.time()
#                print 'time in sift ' + str(time_sift_end - time_sift_start )
                if np.size(kp)>0:
#                   time_dist_compute_start = time.time()
                   histPoints = tH.histogramContour(des,codeBookCenters)
#                   time_dist_compute_end = time.time()
#                   print 'time in dist compute ' + str(time_dist_compute_end - time_dist_compute_start )
#                   time_randomForest_start = time.time()
                   prediction = tC.classify(histPoints, svm)
                   print 'prediction no ' + str(count) + str(prediction)
                   print 'Contour Area of contour no ' + str(count) + str(cv2.contourArea(cnt))

                   cv2.imwrite(str(timeComp) +'Contour' + str(count) + 'Predicted' + str(prediction) + '.png',roiImage)
#                   time_randomForest_end = time.time()
#                   print 'time in random forest ' + str(time_randomForest_end - time_randomForest_start )

                   #print prediction
				## If the predicted class is not the background class then add the prediction to the prediction array and get its centroid
                   if prediction < backgroundClass and prediction > 0:
                       predictionArray.append(prediction)
                       centroidArray.append(centroid)
                       contourArray.append(cv2.contourArea(cnt))
                       labels.append(objects[np.int(prediction)])

#                   time_each_contour_end = time.clock()
#                   print 'time taken for each contour is ' + str(time_each_contour_end - time_each_contour_start)
    newPrediction = []
    newCentroid = []
    newContour = []


    return predictionArray, centroidArray,labels
