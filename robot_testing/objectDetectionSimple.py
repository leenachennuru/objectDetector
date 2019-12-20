#!/usr/bin/env python
"""
@author: 4chennur, 4wahab
"""
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
        #cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        #cv.MoveWindow(self.cv_window_name, 25, 75)

        # And one for the depth image
        #cv.NamedWindow("Depth Image", cv.CV_WINDOW_NORMAL)
        #cv.MoveWindow("Depth Image", 25, 350)

        '''Initialize ros publisher'''
        # topic where we publish
        self.msgPub = rospy.Publisher("vis_imgCoord",
            IC, queue_size = 10)
        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        self.imageSub = rospy.Subscriber("/kinect1_upper/rgb/image_rect_color",
                            Image, self.imageCallback, queue_size=1)
        self.depthSub = rospy.Subscriber("/kinect1_upper/depth/image_rect", Image, self.depthCallback, queue_size=1)
        rospy.loginfo("Waiting for image topics...")

    def imageCallback(self, rosImage):

        global predictionArray, centroidArray,labels


        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(rosImage, "bgr8")
        except CvBridgeError, e:
            print e

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(frame, dtype=np.uint8)

        centroidArray = [self.track(frame), self.track2(frame)]
        labels = ["cube", "ball"]

        # Process the frame using the process_image() function
        #predictionArray, centroidArray,labels = self.processImage(frame)

        # Display the image.
        #cv2.imshow(self.nodeName, frame)

        # Process any keyboard commands
        self.keystroke = cv.WaitKey(5)
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'q':
                # The user has press the q key, so exit
                rospy.signal_shutdown("User hit q key to quit.")



    def depthCallback(self, rosImage):
        global predictionArray, centroidArray
        objectLocations3D = []

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
       # cv2.imshow("Depth Image", depthDisplayImage)
        #depth_display_image = self.process_depth_image(depth_array)

        msg = IC()
        depthDisplayImage = depthDisplayImage[:,:,0];
        if centroidArray != None:
            msg.labelCoord = ()
            for i in range(np.size(centroidArray,0)):
                if centroidArray[i][0] < 0.0:
                    continue;
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

        #print objectLocations3D




        #msg.header.stamp = rospy.Time.now()
        #msg.labelCoord = msg1

        # Publish new image
        self.msgPub.publish(msg)

       #### Create CompressedIamge ###

        #self.subscriber.unregister()


    def track(self, image):

        '''Accepts BGR image as Numpy array
           Returns: (x,y) coordinates of centroid if found
                    (-1,-1) if no centroid was found
                    None if user hit ESC
        '''

        # Blur the image to reduce noise
        blur = cv2.GaussianBlur(image, (5,5),0)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image for only green colors
        lower_green = np.array([120,200,70])
        upper_green = np.array([130,240,124])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Blur the mask
        bmask = cv2.GaussianBlur(mask, (5,5),0)

        # Take the moments to get the centroid
        moments = cv2.moments(bmask)
        m00 = moments['m00']
        centroid_x, centroid_y = None, None
        if m00 != 0:
            centroid_x = int(moments['m10']/m00)
            centroid_y = int(moments['m01']/m00)

        # Assume no centroid
        ctr = (-1,-1)

        # Use centroid if it exists
        if centroid_x != None and centroid_y != None:

            ctr = (centroid_x, centroid_y)
            print "x" + str(centroid_x)
            print "y" + str(centroid_y)

            # Put black circle in at centroid in image
            cv2.circle(image, ctr, 4, (0,0,0))

        # Display full-color image
        #cv2.imshow(self.cv_window_name, image)

        # Force image display, setting centroid to None on ESC key input
        if cv2.waitKey(1) & 0xFF == 27:
            ctr = None

        # Return coordinates of centroid
    	return ctr

    def track2(self, image):

        '''Accepts BGR image as Numpy array
           Returns: (x,y) coordinates of centroid if found
                    (-1,-1) if no centroid was found
                    None if user hit ESC
        '''

        # Blur the image to reduce noise
        blur = cv2.GaussianBlur(image, (5,5),0)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image for only green colors

        # Threshold the HSV image to get only green colors
        #mask = cv2.inRange(hsv, lower_green, upper_green)

        # lower mask (0-10)
        lower_red = np.array([0,140,140])
        upper_red = np.array([5,180,180])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        #upper mask (170-180)
        lower_red = np.array([175,140,140])
        upper_red = np.array([180,180,180])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # Threshold the HSV image to get only green colors

        mask = mask0 + mask1
        # Blur the mask

        # Blur the mask
        bmask = cv2.GaussianBlur(mask, (5,5),0)

        # Take the moments to get the centroid
        moments = cv2.moments(bmask)
        m00 = moments['m00']
        centroid_x, centroid_y = None, None
        if m00 != 0:
            centroid_x = int(moments['m10']/m00)
            centroid_y = int(moments['m01']/m00)

        # Assume no centroid
        ctr = (-1,-1)

        # Use centroid if it exists
        if centroid_x != None and centroid_y != None:

            ctr = (centroid_x, centroid_y)
            print "x" + str(centroid_x)
            print "y" + str(centroid_y)

            # Put black circle in at centroid in image
            cv2.circle(image, ctr, 4, (0,0,0))

        # Display full-color image
        #cv2.imshow(self.cv_window_name, image)

        # Force image display, setting centroid to None on ESC key input
        if cv2.waitKey(1) & 0xFF == 27:
            ctr = None

        # Return coordinates of centroid
        return ctr


    def processImage(self, frame):
        predictionArray, centroidArray,labels = tS.processAndClassify(frame)
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
        #objectDetection.processCurImage()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
