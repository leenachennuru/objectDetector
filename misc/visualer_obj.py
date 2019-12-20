#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""


# -*- coding: utf-8 -*-

import rospy
import rospkg
import sys
import cv2
import cv
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import testSegmentedThreshold as tS
from phri_common_msgs.msg import ImgCoordArray as IC
from phri_common_msgs.msg import ImgCoordinates as IM
import random

#roslib.load_manifest('person_detection')
#
##------------------------------------------------------------------------------
## Message formats
#from sensor_msgs.msg import CompressedImage
#from person_detection.msg import FaceCoordinatesArray
#------------------------------------------------------------------------------
# Own Libraries or Scripts
#******************************************************************************


#******************************************************************************
# MODULE
#------------------------------------------------------------------------------
class Visualizer:
    '''
    Visualizer for the face detection results.
    '''


    def __init__(self):
        '''
        Initializing variables and setting up the ROS communication (creating
        the ROS node and registering shutdown hook, which is the function that
        is executed when the ROS node is shutdown).
        '''
        # Variables
        self.imageFrame = None
        self.IC = []

        # Initializing ROS node
        name='visualizer_obj'+str(random.randint(0,100))
        rospy.init_node(name)
        # anonymous=True

        # Creating the cv_bridge object
        self.bridge = CvBridge()

        # Running the ROS node
        self.setup_ros_node()

        # Registering shotdown hook for function to be executed on shutdown
        rospy.on_shutdown(self.clean_up)

        # Don't stop until ROS shutdown
        while not rospy.is_shutdown():
            if self.imageFrame != None:
                self.visualize()
            self.rate.sleep()

    #end of function


    def setup_ros_node(self):
        '''
        Subscribes to the topic providing the RGB camera images and ... TODO.
        '''
        # Creating ROS subscriber for rgb camera images from the Xtion
        self.rgbCameraSubscriber = rospy.Subscriber(
            '/camera/rgb/image_rect_color',
            CompressedImage,
            self.rgb_camera_callback)

        # Creating ROS subscriber for person coordinates
        self.objCoordinatesSubscriber = rospy.Subscriber(
            'vis_imgCoord', IC,
            self.obj_coordinates_callback)

        # Achieves looping in desired rate
        self.rate = rospy.Rate(10) # 10hz

        # Printing status
        rospy.loginfo('ROS communication running')
    #end of function


    def rgb_camera_callback(self, data):
        '''
        Callback function handling the received RGB camera images. Converts
        image to the OpenCV format, detects closest face and publishes pixel
        coordinates for that face.
        '''
        # Converting the data to OpenCV format
        rgbImage = np.fromstring(data.data, np.uint8)

        # Converting the frame to a numpy array
        self.imageFrame = cv2.imdecode(rgbImage, cv2.CV_LOAD_IMAGE_COLOR)
    #end of function


    def obj_coordinates_callback(self, data):
        '''
        Updates the face coordinates array upon receiving new topic message.
        '''
        self.IC = data.array
    #end of function


    def visualize(self):
        '''
        Visualizes the face detection results. Each face centroid is indicated
        by a blue dot.
        '''
        displayImage = np.copy(self.imageFrame)

        color = (255, 0, 0)

        # Drawing rectangles around the detected persons
        for obj in self.IC:
                #center = (int(obj.x), int(obj.y))
                #cv2.rectangle(displayImage, center, 2, color, 1)
                cv2.rectangle(displayImage,(int(obj.x - 20), int(obj.y - 20)),(int(obj.x + 20), int(obj.y + 20)),(0,255,0),3)

        # Displaying RGB image array
        cv2.imshow('Object Detection Results', displayImage)
        cv2.waitKey(1)

        self.IC = []


    def clean_up(self):
        '''
        Cleans up.
        '''
        rospy.loginfo('Shutting down...')
        cv2.destroyAllWindows()
    #end of function


# end of module
#******************************************************************************


#******************************************************************************
# FUNCTIONS
#------------------------------------------------------------------------------
def main():
    '''
    Runs face detection.
    '''
    try:
        visualizer = Visualizer()
    except rospy.ROSInterruptException:
        pass

# end of function
#******************************************************************************


#******************************************************************************
# MAIN
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#******************************************************************************
