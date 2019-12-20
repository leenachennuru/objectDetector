# -*- coding: utf-8 -*-

import os
import shutil
import cv2
import numpy


"""Separate the files in sequences of N frames"""

frames = 4
directory = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/imagesRaw/"

directorySave = "/export/experiments/emotionalFeedback/cam3D_"+str(frames)+"/"

for classes in sorted(os.listdir(directory)):
    
    currentSequenceNumber = 0
    print "Class:", classes
    for sequence in sorted(os.listdir(directory+"/"+classes+"/")):             
        currentSequenceImages = 0
        #imageDirectory = sorted(os.listdir(directory+"/"+classes+"/"+sequence+"/"), key=lambda x: int(x.split('.')[0]))
        imageDirectory = os.listdir(directory+"/"+classes+"/"+sequence+"/images/")
        #imageDirectory = imageDirectory[4:len(imageDirectory)-4]
        #print "Here:", imageDirectory
        lastImage = ""
        #audioFiles = os.listdir(directoryAudio+"/"+classes)
        #currentAudioFile = audioFiles[currentAudio]  
        
        firstImage = True
        for image in  imageDirectory:
                
            if currentSequenceImages >= frames:
                currentSequenceNumber = currentSequenceNumber+1
                currentSequenceImages = 0                    
                
            copyFrom = directory+"/"+classes+"/"+sequence+"/images/"+"/"+image
            copyTo = directorySave+"/"+classes+"/"+classes+str(currentSequenceNumber)+"/"            
            if not os.path.exists(copyTo): os.makedirs(copyTo)
            shutil.copyfile(copyFrom,copyTo+"/"+image)   
            currentSequenceImages = currentSequenceImages+1
            lastImage = copyFrom,copyTo+"/"
        
        
        while currentSequenceImages < 4:
            shutil.copyfile(lastImage[0],lastImage[1]+str(currentSequenceImages)+".png")   
            currentSequenceImages = currentSequenceImages+1
            
        currentSequenceNumber = currentSequenceNumber+1