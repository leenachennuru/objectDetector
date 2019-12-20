# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs
CompressedImage. It converts the CompressedImage into a numpy.ndarray,
then detects and marks features in that image. It finally displays
and publishes the new image - again as CompressedImage topic.
"""

import roslib; #roslib.load_manifest('rbx_vision')
import rospy
import sys
import cv2
import cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import testSegmentedPro as tS
import detectorDescriptor2 as dd
import preObj as prePro



# Python libs
import sys, time

# numpy and scipy
from scipy.ndimage import filters

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/camera/image_new/compressed",
            CompressedImage, queue_size = 10)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/rgb/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to /camera/rgb/image_raw/compressed"


    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

        #### Feature detectors using CV2 ####
        # "","Grid","Pyramid" +
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        #method = "FAST"
        #feat_det = cv2.FeatureDetector_create(method)


        #convert np image to grayscale
#        grey_imagenp = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#        print 'The image is being processed'

        imageKey = tS.processAndClassify(image_np)
        #featPoints,des,keyImage = dd.featureDetectDesORB(grey_imagenp)
#
#        for featpoint in featPoints:
#            x,y = featpoint.pt
#            cv2.circle(image_np,(int(x),int(y)), 3, (0,0,255), -1)
#

        print 'The image is being processed'
        cv2.imshow('cv_img', imageKey)
        cv2.waitKey(2)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        #self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
