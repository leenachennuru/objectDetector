# -*- coding: utf-8 -*-
import os
import cv2
import math

directory = "/export/pablo/Datasets/SFEW/training"
directorySave = "/export/pablo/Datasets/SFEW/training_faces_resized/"


def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        img = cv2.resize(img, (300,200)) 
        
        im = cv2.equalizeHist(img)
        side = math.sqrt(im.size)
        minlen = int(side / 20)
        maxlen = int(side / 2)
        flags = 1
        cc = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml")
    
        
        #cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml")
        #rects = cascade.detectMultiScale(img, 1.1, 4, 1, (20,20))
        
        rects = cc.detectMultiScale(im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
    
        if len(rects) == 0:            
            return None
        rects[:, 2:] += rects[:, :2]
        
        return box(rects,img2)

def box(rects, img):        
        for x1, y1, x2, y2 in rects:
            
            #cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
            img = img[y1:y2, x1:x2]
            
            #newx,newy = 28,28 #new size (w,h)
            #newimage = cv2.resize(img2,(newx,newy))
            #cv2.imwrite(path, img2);
            return img
            
for classe in os.listdir(directory):            
    for image in os.listdir(directory+"/"+classe):
        img = cv2.imread(directory+"/"+classe+"/"+image)
        face = detectFace(img)
        if not img  == None:
            saveIn = directorySave + "/"+classe
            if not os.path.exists(saveIn):
                os.makedirs(saveIn)
            cv2.imwrite(saveIn+"/"+image, face)
    
    
    
    
    

