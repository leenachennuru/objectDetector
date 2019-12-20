# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""
#roslib.load_manifest('rbx_vision')
import rospy
import rospkg
import sys
import cv2
import cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import testSegmented as tS
from phri_common_msgs.msg import ImgCoordArray as IC
from phri_common_msgs.msg import ImgCoordinates as IM
from sklearn.externals import joblib

global codeBookCenters
global svm

codeBookCenters = None
svm = None


def processImage(rosImage,codeBookCenters,svm):
    global predictionArray, centroidArray,labels
    frame = np.array(rosImage, dtype=np.uint8)
        # Process the frame using the process_image() function
    predictionArray, centroidArray,labels = tS.processAndClassify(frame,codeBookCenters,svm)
    return predictionArray, centroidArray,labels


def getDepth(image,x,y):
    try:
        return image[y][x]
    except:
        print "Error occured in getDepth...\n"
        return -1

class objectDetection():
    def __init__(self):
        self.nodeName = "object_detection"

        rospy.init_node(self.nodeName)

        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()

        self.imageSub = rospy.Subscriber("/camera/rgb/image_rect_color",
                            Image, self.imageCallback, queue_size=1)
        self.depthSub = rospy.Subscriber("/camera/depth/image_rect", Image, self.depthCallback, queue_size=1)
        rospy.loginfo("Waiting for image topics...")

    def imageCallback(self, rosImage):
        global rgb_frame
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rosImage, "bgr8")
        except CvBridgeError, e:
            print e

    def depthCallback(self,rosImage):
        global depth_frame
        try:
            depth_frame = self.bridge.imgmsg_to_cv2(rosImage, "16UC1")
            # The depth image is a single-channel float32 image
        except CvBridgeError, e:
            print e

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()


def main(args):
    global depth_frame, rgb_frame
    predictionArray = None
    centroidArray = None
    labels = None
    rgb_frame = None
    depth_frame = None

    #cv.NamedWindow('rgb_frame', cv.CV_WINDOW_NORMAL)
    #cv.MoveWindow('rgb_frame', 25, 75)
    while True:
        try:
            objectDetection()
            msgPub = rospy.Publisher("vis_imgCoord",IC, queue_size = 10)
            while rgb_frame != None:
		cv.NamedWindow('rgb_frame', cv.CV_WINDOW_NORMAL)
   		cv.MoveWindow('rgb_frame', 25, 75)
                cv2.imshow('rgb_frame',rgb_frame)
#                cv2.waitkey(0)
                predictionArray, centroidArray,labels = processImage(rgb_frame,codeBookCenters,svm)
                msg = IC()
                depthArray = np.array(depth_frame, dtype=np.float32)
                depthDisplayImage = cv2.normalize(depthArray, depthArray, 0, 1, cv2.NORM_MINMAX)
                depthDisplayImage = depthDisplayImage[:,:,0];
                if centroidArray != None:
                    msg.labelCoord = ()
                    for i in range(np.size(centroidArray,0)):
                        msg1 = IM
                        depthValue =  getDepth(depthDisplayImage,centroidArray[i][0],centroidArray[i][1])
                        msg1.label = labels[i]
                        msg1.x = centroidArray[i][0]
                        msg1.y = centroidArray[i][1]
                        msg1.z = depthValue
                        msg.stamp = rospy.Time.now()
                        msg.labelCoord = msg.labelCoord + (msg1,)
                        msgPub.publish(msg)
        except KeyboardInterrupt:
            print "Shutting down vision node."
            cv.DestroyAllWindows()

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    path_to_package = rospack.get_path("object_recognition")
    codeBookCenterspath = "/Scripts/CodeBook/siftSubsampledCenters75000.npy"

    codeBookCenters = np.load(path_to_package + codeBookCenterspath)
    print 'codebook is loaded'
    #Uncomment for opencv SVM
    #svm = cv2.SVM()
    #Scikit-learn svm implementation
    svm = joblib.load(path_to_package + "/Scripts/RandomForestsift75000.pkl")
    print 'svm is loaded'
    main(sys.argv)
