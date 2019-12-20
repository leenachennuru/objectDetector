# -*- coding: utf-8 -*-

import cv2
import os
import math

def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        
        cascade = cv2.CascadeClassifier("/informatik2/wtm/home/barros/Workspace/faceDetection/haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(img, 1.2, 4, 1, (20,20))
    
        if len(rects) == 0:            
            return None
        rects[:, 2:] += rects[:, :2]
        
        return rects
        return box(rects,img2, img2)

def box(rects, img, img2):        
        imgs = []
        for x1, y1, x2, y2 in rects:            
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 255, 0), 2)            
            imgs.append(img[y1:y2, x1:x2])
                        
        return imgs, img2


vc = cv2.VideoCapture(0) 
#vc.set(3,1024)
#vc.set(4,768)

if vc.isOpened(): # try to get the first frame
    rval, f = vc.read()
 #   rval2, f2 = vc2.read()
else:
    rval = False
    
lookingTo = (0,0)

searchingForAFace = True
faceTracked = False

lastFrames = []

while rval:  
    
    rects = detectFace(f)
    if len(lastFrames) > 3:
        del lastFrames[0]
        print "erase"
        
    if not rects == None:     
        faceTrackedInThisFrame = False               
        for x1, y1, x2, y2 in rects:      
            cv2.rectangle(f, (x1, y1), (x2, y2), (192, 19, 19), 2)  
            lookTo = ((x2 + x1)/2), ((y2+y1)/2) 
            if searchingForAFace:
                lookingTo = lookTo
                lastFrames.append(lookingTo)
                searchingForAFace = False                
                faceTrackedInThisFrame = True
                cv2.rectangle(f, (x1, y1), (x2, y2), (19, 71, 192), 2) 
                #Robot, look to the person
            else:
                distance = 1000000
                for lookToPast in lastFrames:                
                    distanceNow = math.hypot(lookTo[0] - lookToPast[0], lookTo[1] - lookToPast[1])
                    if distanceNow < distance:
                        distance = distanceNow
                        
                if distance < 50:
                    lookingTo = lookTo             
                    lastFrames.append(lookingTo)
                    faceTrackedInThisFrame = True
                    cv2.rectangle(f, (x1, y1), (x2, y2), (19, 71, 192), 2) 
                    #Robot, look to the person
                    
        if not faceTrackedInThisFrame:
            lookingTo = (0,0)
            searchingForAFace = True        
            lastFrames = []
            
#            cv2.circle(f, lookTo, 10, (19, 71, 192), thickness=-1)                                                                   
#            cv2.rectangle(f, (x1, y1), (x2, y2), (19, 71, 192), 2) 
            
    cv2.imshow("FaceLiveImage", f)                 
    rval, f = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break        