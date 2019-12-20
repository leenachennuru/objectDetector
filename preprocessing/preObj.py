"""
@author: 4chennur, 4wahab
"""
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import misc as msc
import cv
import sys
#from skimage.segmentation import slic

sys.path.append('../../../')
#from phri_common_msgs.msg import ImgCoordinates



def unit8Image(imageGrayInput):
    try:
        imageGrayInput_unit8 = np.uint8(imageGrayInput)
        return imageGrayInput_unit8
    except:
        print 'Make sure numpy is imported and is renamed as np. Also make sure your input is a greyscale image'

def otsuBin(imageGrayInput):
    imageGrayInput_unit8 = unit8Image(imageGrayInput)
    try:
        ret, thresh = cv2.threshold(imageGrayInput_unit8, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh
    except TypeError:
        print 'The input image in the function otsuBin is not of data type unit8 or its not grayscale. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in otsuBin is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function otsubin is not working. You have most likely given invalid arguments.'


def meanShift(imageInput):
    imageInput_unit8 = unit8Image(imageInput)
    try:
        meanShifted = cv2.pyrMeanShiftFiltering(imageInput_unit8, 20, 20)
        return meanShifted
    except TypeError:
        print 'The input image in the function meanShift is not of data type unit8. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in meanShift is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function meanShift is not working. You have most likely given invalid arguments.'



def cannyEdge(imageInput):
    ''' Canny edge detctor on an Input image which should be in unit8 format
    This returns the edgeDetetction variable which contains the image with the edges marked out '''
    imageInput_unit8 = unit8Image(imageInput)
    try:
        edgeDetection = cv2.Canny(imageInput_unit8, 10, 10)
        return edgeDetection
    except TypeError:
        print 'The input image in the function cannyEdge is not of data type unit8. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in cannyEdge is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function cannyEdge is not working. You have most likely given invalid arguments.'


def adapThresh(imageInput):
    imageInput_unit8 = unit8Image(imageInput)
    try:
        threshAdaptive = cv2.adaptiveThreshold(imageInput, 255, 1, 1, 11, 2)
        return threshAdaptive
    except TypeError:
        print 'The input image in the function adapThresh is not of data type unit8. Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in adapThresh is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function adapThresh is not working. You have most likely given invalid arguments.'



def contourFind(prepImage):
    prepImage_unit8 = unit8Image(prepImage)
    try:
        contours, hierarchy = cv2.findContours(prepImage_unit8, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    except TypeError:
        print 'The input image in the function contourFind is not binary. Plese convert the image to binary before using this function.'
    except IOError:
        print 'The path to the file in contourFind is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function contourFind is not working. You have most likely given invalid arguments.'

def contourFindFull(prepImage):
    contours, hierarchy = cv2.findContours(prepImage, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy



def contourDraw(imageInput, prepImage):
    prepImage_unit8 = unit8Image(prepImage)
    imageInput_unit8 = unit8Image(imageInput)
    try:
        contours, hierarchy = cv2.findContours(prepImage_unit8, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        imageInputCopy = imageInput_unit8.copy()
        cv2.drawContours(imageInputCopy, contours, -1, (255, 0, 0), 3)
        return imageInputCopy
    except TypeError:
        print 'The input image in the function contourDraw is not . Plese convert the image to unit8 using numpy.unit8.'
    except IOError:
        print 'The path to the file in contourDraw is not correctly specified. Please check that the file is in the correct location.'
    else:
        print 'The function contourDraw is not working. You have most likely given invalid arguments.'

#def superPixelSeg(image):
 #   numSegments = 20
  #  segments = slic(image, n_segments = numSegments, sigma = 5)
   # return segments


##Testing Phase
