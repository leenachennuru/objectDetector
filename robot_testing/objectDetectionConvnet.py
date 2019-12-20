# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

#!/usr/bin/env python

import roslib; #roslib.load_manifest('rbx_vision')
import rospy
import rospkg
import sys
import cv2
import cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import testSegmented as tS
from phri_common_msgs.msg import ImgCoordArray as IC
from phri_common_msgs.msg import ImgCoordinates as IM
from sklearn.externals import joblib

############### Convnet imports ######################
from convNet.mccnn.Utils import DataUtil
from convNet.mccnn.Networks import MCCNN
import convNet.mccnn.MCCNNExperiments
import cv2
import os
import glob

global networkTopology, network,trainingParameters

networkTopology = None
network = None
trainingParameters = None

predictionArray = None
centroidArray = None
labels = None



class objectDetection():
    def __init__(self):
        self.nodeName = "object_detection"

        rospy.init_node(self.nodeName)

        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.nodeName
        cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        cv.MoveWindow(self.cv_window_name, 25, 75)

        # And one for the depth image
        cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Depth Image", 25, 350)

        '''Initialize ros publisher'''
        # topic where we publish
        self.msgPub = rospy.Publisher("vis_imgCoord",
            IC, queue_size = 10)
        # Create the cv_bridge object
        self.bridge = CvBridge()
        self.networkTopology =  networkTopology
        self.network = network
        self.trainingParameters = trainingParameters

        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        self.imageSub = rospy.Subscriber("/camera/rgb/image_rect_color",
                            Image,  self.imageCallback, queue_size=1)
        self.depthSub = rospy.Subscriber("/camera/depth/image_rect", Image, self.depthCallback, queue_size=1)
        rospy.loginfo("Waiting for image topics...")

    def imageCallback(self, rosImage):
        global predictionArray, centroidArray,labels
        # Use cv_bridge() to convert the ROS image to OpenCV format
        print 'started image callback'
        try:
            frame = self.bridge.imgmsg_to_cv2(rosImage, "bgr8")
        except CvBridgeError, e:
            print e



        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(frame, dtype=np.uint8)

        # Process the frame using the process_image() function
        predictionArray, centroidArray,labels = self.processImage(frame, self.networkTopology, self.network,self.trainingParameters)

        # Display the image.
        cv2.imshow(self.nodeName, frame)

        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")


    def depthCallback(self, rosImage):
        global predictionArray, centroidArray

        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            depthImage = self.bridge.imgmsg_to_cv2(rosImage, "16UC1")

            # The depth image is a single-channel float32 image

        except CvBridgeError, e:
            print e


        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        depthArray = np.array(depthImage, dtype=np.float32)

        # Normalize the depth image to fall between 0 (black) and 1 (white)
        depthDisplayImage = cv2.normalize(depthArray, depthArray, 0, 1, cv2.NORM_MINMAX)

        # Display the result
        cv2.imshow("Depth Image", depthDisplayImage)
        #depth_display_image = self.process_depth_image(depth_array)

        msg = IC()
        depthDisplayImage = depthDisplayImage[:,:,0];
        if centroidArray != None:
            msg.labelCoord = ()
            for i in range(np.size(centroidArray,0)):
                msg1 = IM
                depthValue =  self.getDepth(depthDisplayImage,centroidArray[i][0],centroidArray[i][1])
                msg1.label = labels[i]
                msg1.x = centroidArray[i][0]
                msg1.y = centroidArray[i][1]
                msg1.z = depthValue
                #pixelLocation3D = (labels[i],centroidArray[i][0],centroidArray[i][1],depthValue)
                #objectLocations3D.append(pixelLocation  #
                msg.stamp = rospy.Time.now()
                msg.labelCoord = msg.labelCoord + (msg1,)

        self.msgPub.publish(msg)


    def processImage(self, frame, networkTopology, network,trainingParameters):
        predictionArray, centroidArray,labels = tS.processAndClassify(frame, networkTopology,network,trainingParameters)
        return predictionArray, centroidArray,labels

    def processDepthImage(self, frame):
        # Just return the raw image for this demo
        return frame

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()

            ### Create CompressedIamge ####
    def getDepth(self,image,x,y):
        try:
            return image[y][x]
        except:
            print "Error occured in getDepth...\n"
            return -1


def main(args):
    try:
        objectDetection()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    path_to_package = rospack.get_path("object_recognition")
    modelDirectoryCPUpath = "/Scripts/convNet/experiments/convNetContoursResized20Epoch/model/repetition_0_BestTest_convNetContoursResized20Epoch_.save"

    modelDirectoryCPU = np.load(path_to_package + modelDirectoryCPUpath)

    networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectoryCPU)

    experimentParameters[0] = os.path.dirname(os.path.abspath(modelDirectoryCPU))
    experimentParameters.append(False)

    saveNetworkParameters = [False]
    network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )

    main(sys.argv)
