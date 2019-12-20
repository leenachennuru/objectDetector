# -*- coding: utf-8 -*-

import cv2
import os
import scipy

print scipy.__version__

subject = "Subject1"
saveDirectory = "/export/experiments/dynamicGestures"

gestures = ["Circle", "Hello", "PointL", "PointR", "Stop", "Turn"]

repetitionsPerGesture = 10

vc = cv2.VideoCapture(0) 

if vc.isOpened(): # try to get the first frame
    rval, f = vc.read()
 #   rval2, f2 = vc2.read()
else:
    rval = False
    
    
while rval:  
    for gesture in gestures:
        for repetition in range(repetitionsPerGesture):
            imageNumber = 0
            while imageNumber < 100:
                rval, f = vc.read()
                saveIn = saveDirectory +"/"+subject+"/"+gesture+"/"+"Sequence_"+str(repetition)+"/"
                if not os.path.exists(saveIn) : os.makedirs(saveIn)
                cv2.imwrite(saveIn)
                imageNumber = imageNumber+1
            raw_input("Gesture:", gesture, " - Repetition:", repetition)    
                        
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0) 