#!/usr/bin/env python
"""
@author: 4chennur, 4wahab
"""
#import roslib; #roslib.load_manifest('rbx_vision')
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
#from phri_common_msgs.srv import ImgCoordArray as IC
from sklearn.externals import joblib

global codeBookCenters
global svm

codeBookCenters = None
svm = None

global CoordinateArray
CoordinateArray = []

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
        #cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        #cv.MoveWindow("Depth Image", 25, 350)

        '''Initialize ros publisher'''
        # topic where we publish
        self.msgPub = rospy.Publisher("vis_imgCoord",
            IC, queue_size = 10)
        # Create the cv_bridge object
        self.bridge = CvBridge()
        self.codeBookCenters =  codeBookCenters
        self.svm = svm

        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        self.imageSub = rospy.Subscriber("/kinect1_upper/rgb/image_rect_color/compressed",
                            CompressedImage,  self.imageCallback, queue_size=1)
        #self.depthSub = rospy.Subscriber("/camera/depth/image_rect", Image, self.depthCallback, queue_size=1)
        rospy.loginfo("Waiting for image topics...")

    def imageCallback(self, rosImage):
        msg = IC()
        msg.stamp = rospy.Time.now()
        global CoordinateArray
        global predictionArray, centroidArray,labels
        # Use cv_bridge() to convert the ROS image to OpenCV format
        print 'started image callback'
#        try:
#            frame = self.bridge.imgmsg_to_cv2(rosImage, "bgr8")
#        except CvBridgeError, e:
#            print e

        np_arr = np.fromstring(rosImage.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        cv2.imshow(self.nodeName, frame)

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        #frame = np.array(frame, dtype=np.uint8)

        # Process the frame using the process_image() function
        predictionArray, centroidArray,labels = self.processImage(frame, self.codeBookCenters, self.svm)

        # Display the image.

        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")

        #depthDisplayImage = depthDisplayImage[:,:,0];
        if centroidArray != None:
            msg.labelCoord = ()
            for i in range(np.size(centroidArray,0)):
                msg1 = IM()
                #depthValue =  self.getDepth(depthDisplayImage,centroidArray[i][0],centroidArray[i][1])
                msg1.label = labels[i]
                msg1.x = centroidArray[i][0]
                msg1.y = centroidArray[i][1]
                msg1.z = 0
                #pixelLocation3D = (labels[i],centroidArray[i][0],centroidArray[i][1],depthValue)
                #objectLocations3D.append(pixelLocation  #
                msg.labelCoord = msg.labelCoord + (msg1,)
		CoordinateArray = msg
        #print objectLocations3D




        #msg.header.stamp = rospy.Time.now()
        #msg.labelCoord = msg1

        # Publish new image
        self.msgPub.publish(msg)


#    def depthCallback(self, rosImage):
#        global predictionArray, centroidArray
#        #objectLocations3D = []
#
#        # Use cv_bridge() to convert the ROS image to OpenCV format
#        try:
#            depthImage = self.bridge.imgmsg_to_cv2(rosImage, "16UC1")
#
#            # The depth image is a single-channel float32 image
#
#        except CvBridgeError, e:
#            print e
#
#
#        # Convert the depth image to a Numpy array since most cv2 functions
#        # require Numpy arrays.
#        depthArray = np.array(depthImage, dtype=np.float32)
#
#        # Normalize the depth image to fall between 0 (black) and 1 (white)
#        depthDisplayImage = cv2.normalize(depthArray, depthArray, 0, 1, cv2.NORM_MINMAX)
#
#        # Display the result
#        cv2.imshow("Depth Image", depthDisplayImage)
#        #depth_display_image = self.process_depth_image(depth_array)
#
#        msg = IC()
#        depthDisplayImage = depthDisplayImage[:,:,0];
#        if centroidArray != None:
#            msg.labelCoord = ()
#            for i in range(np.size(centroidArray,0)):
#                msg1 = IM()
#                depthValue =  self.getDepth(depthDisplayImage,centroidArray[i][0],centroidArray[i][1])
#                msg1.label = labels[i]
#                msg1.x = centroidArray[i][0]
#                msg1.y = centroidArray[i][1]
#                msg1.z = depthValue
#                #pixelLocation3D = (labels[i],centroidArray[i][0],centroidArray[i][1],depthValue)
#                #objectLocations3D.append(pixelLocation  #
#                msg.stamp = rospy.Time.now()
#                msg.labelCoord = msg.labelCoord + (msg1,)
#
#        #print objectLocations3D
#
#
#
#
#        #msg.header.stamp = rospy.Time.now()
#        #msg.labelCoord = msg1
#
#        # Publish new image
#        self.msgPub.publish(msg)
#
#       #### Create CompressedIamge ###
#
#        #self.subscriber.unregister()
#




    def processImage(self, frame, codeBookCenters, svm):
        predictionArray, centroidArray,labels = tS.processAndClassify(frame, codeBookCenters, svm)
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


#def sendCoordinates(boolean):#
#    if boolean == True:
#        return CoordinateArray
#    else:
#        return None

#def objectDetectionServer():
#    # Console Info
#    rospy.loginfo('Object Detection Service Initializing')
#
#    rospy.init_node('objectDetection_server', anonymous=False)
#    serviceObjectDetection = rospy.Service('vis_imgCoord', IC, sendCoordinates)
#
#    print "Server has started"

def main(args):
    try:
        objectDetection()
        #objectDetectionServer()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    path_to_package = rospack.get_path("object_recognition")
    codeBookCenterspath = "/Scripts/CodeBook/siftSubsampledCenters10000.npy"

    codeBookCenters = np.load(path_to_package + codeBookCenterspath)
    print 'codebook is loaded'
    #Uncomment for opencv SVM
    #svm = cv2.SVM()
    #Scikit-learn svm implementation
    svm = joblib.load(path_to_package + "/Scripts/SVM/RandomForestsift10000.pkl")
    print 'svm is loaded'
    main(sys.argv)
