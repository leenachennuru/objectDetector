# -*- coding: utf-8 -*-
import os
import numpy
import cv2
import shutil

def doConvShadow(images):

        
        total = []
        
        binaries = []        
        
        for i in range(len(images)):                      
            img4 = numpy.asarray(images[i])            
           
            if i == 0:
                previous = img4
            else:
                
                newImage = cv2.absdiff(img4, previous)
                
                binaries.append(newImage)
          

                
        for i in range(len(binaries)):            
           
            weight = float(i)/float(len(binaries))
            
            
            img4 = numpy.asarray(binaries[i]) * weight
            #img4 = numpy.asarray(binaries[i])
             
            if(i==0):
                total = img4
                
            else:
                #total = cv2.absdiff(img4, total)
                total = total + (img4)
                
        return total
        
        
#"""Transform the sequence in N movement representations"""  
#
#frames = 10
#source =  "/data/datasets/FABO/FABO/"      
#destination = "/data/datasets/FABO/FABO_Apex_"+str(frames)+"/"   
#
##separationLimit = {"Anger":25, "Contempt":10, "Disgust":36, "Fear":15, "Happy":44, "Neutral":185, "Sadness":15, "Surprise":50}
#separationLimit = {"Anger":500, "Contempt":500, "Disgust":500, "Fear":500, "Happy":500, "Neutral":500, "Sadness":500, "Surprise":500}
#
#
#fileNumber = 0
#neutralSequence = 0
#numberOfNeutralSequences = 0
#for c in os.listdir(source):
#    separation = "training"
#    numberOfSequences = 0    
#    for s in os.listdir(source+"/"+c):       
#        print "Folder:", source+"/"+c+"/"+s         
#        files = os.listdir(source+"/"+c+"/"+s)
#        if ".DS_Store" in files:
#            files.remove(".DS_Store")
#            
#        files = sorted(files, key=lambda x: int(x.split('.')[0]))
#            
#        imageIndex = 0
#        numberOfSequenceImages = len(files)/frames
#        
#        for a in range(numberOfSequenceImages):
#            imagesInsequence = []
#            for ah in range(frames):
#                imagesInsequence.append(cv2.imread(source+"/"+c+"/"+s+"/"+files[imageIndex]))
#                imageIndex = imageIndex+1
#            
#            shadowImage = doConvShadow(imagesInsequence)
#            copyTo = destination+"/"+separation+"/"+c+"/"+s+"/"
#                
#            if not os.path.exists(copyTo): os.makedirs(copyTo)
#            
#            copyTo = copyTo+"/"+str(fileNumber)+".png"
#            cv2.imwrite(copyTo,shadowImage)
#            fileNumber = fileNumber+1
# 

"""Separate the files in sequences of N frames"""

frames = 3
directory = "/data/datasets/FABO/FABO_Movement_10Frames/"

directorySave = "/data/datasets/FABO/FABO_M10_"+str(frames)+"_Frames/"

for classes in sorted(os.listdir(directory)):
    
    currentSequenceNumber = 0
    print "Class:", classes
    for sequence in sorted(os.listdir(directory+"/"+classes+"/")):             
        currentSequenceImages = 0
        #imageDirectory = sorted(os.listdir(directory+"/"+classes+"/"+sequence+"/"), key=lambda x: int(x.split('.')[0]))
        imageDirectory = os.listdir(directory+"/"+classes+"/"+sequence)
        imageDirectory = sorted(imageDirectory, key=lambda x: int(x.split('.')[0]))
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
                
            copyFrom = directory+"/"+classes+"/"+sequence+"/"+image
            copyTo = directorySave+"/"+classes+"/"+str(currentSequenceNumber)+"/"            
            if not os.path.exists(copyTo): os.makedirs(copyTo)
            shutil.copyfile(copyFrom,copyTo+"/"+image)   
            currentSequenceImages = currentSequenceImages+1
            lastImage = copyFrom,copyTo+"/"
        
#        
        while currentSequenceImages < frames:
            shutil.copyfile(lastImage[0],lastImage[1]+str(currentSequenceImages)+".png")   
            currentSequenceImages = currentSequenceImages+1
            
        currentSequenceNumber = currentSequenceNumber+1                           