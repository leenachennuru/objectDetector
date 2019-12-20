# -*- coding: utf-8 -*-

import os
import shutil
import cv2
import numpy


def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        
        cascade = cv2.CascadeClassifier("/informatik2/wtm/home/barros/Workspace/faceDetection/haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(img, 1.3, 4, 1, (20,20))
    
        if len(rects) == 0:            
            return img2
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
            

          
 
"""Extract the N last frames of each sequence FACE"""  

frames = 4
source =  "/data/datasets/Cohn-Kanade/CKFaces/"      
destination = "/data/datasets/Cohn-Kanade/CKFaces_Frames_"+str(frames)+"/"   

#separationLimit = {"Anger":25, "Contempt":10, "Disgust":36, "Fear":15, "Happy":44, "Neutral":185, "Sadness":15, "Surprise":50}
separationLimit = {"Anger":500, "Contempt":500, "Disgust":500, "Fear":500, "Happy":500, "Neutral":500, "Sadness":500, "Surprise":500}


fileNumber = 0
neutralSequence = 0
numberOfNeutralSequences = 0
for c in os.listdir(source):
    separation = "training"
    numberOfSequences = 0    
    for s in os.listdir(source+"/"+c):       
        print "Folder:", source+"/"+c+"/"+s         
        files = os.listdir(source+"/"+c+"/"+s)
        if ".DS_Store" in files:
            files.remove(".DS_Store")
            
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        
        
        numberOfSequences = numberOfSequences+1
        if numberOfSequences >separationLimit[c]:
            print "Class:", c
            print "SequenceNumber:", numberOfSequences
            print "SeparationLimit:", separationLimit[c]
            separation = "test"
        else:
            separation = "training"
            
        for a in range(frames):
            
            copyFrom = source+"/"+c+"/"+s+"/"+files[len(files)-a-1]
            copyTo = destination+"/"+separation+"/"+c+"/"+s+"/"
                
            if not os.path.exists(copyTo): os.makedirs(copyTo)
                    
        
            copyTo = copyTo+"/"+str(fileNumber)+".png"
            print "Copy From:", copyFrom       
            print "Copy To:", copyTo
            print "-------------------"
            #cv2.imwrite(copyTo, img)     
            shutil.copyfile(copyFrom,copyTo) 
            fileNumber = fileNumber+1
        neutralSequence = neutralSequence+1

        
        numberOfNeutralSequences = numberOfNeutralSequences+1
        separation = "training"
        if numberOfNeutralSequences >separationLimit["Neutral"]:
            separation = "test"
        
        for a in range(frames):
            copyFromNeutral = source+"/"+c+"/"+s+"/"+files[a]
            copyToNeutral = destination+"/"+separation+"/Neutral"+"/"+"/"+str(neutralSequence)+"/"
            if not os.path.exists(copyToNeutral): os.makedirs(copyToNeutral)
            copyToNeutral = copyToNeutral+"/"+str(fileNumber)+".png"
            
            img = detectFace(cv2.imread(copyFromNeutral)) 
            cv2.imwrite(copyToNeutral, img)  
            #shutil.copyfile(copyFromNeutral,copyToNeutral)  
            fileNumber = fileNumber+1
            
           
#"""Extract the face of all CK+. Static Images"""         
#source =  "/data/datasets/Cohn-Kanade/Cohn-Kanade+/"      
#destination = "/data/datasets/Cohn-Kanade/CKFaces/"   
#
#fileNumber = 0
#for c in os.listdir(source):
#    for s in os.listdir(source+"/"+c):       
#        print "Folder:", source+"/"+c+"/"+s         
#        files = os.listdir(source+"/"+c+"/"+s)
#        if ".DS_Store" in files:
#            files.remove(".DS_Store")
#        files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[2]))
#        
#        for i in files:
#            copyFrom = source+"/"+c+"/"+s+"/"+i
#            copyTo = destination+"/"+c+"/"+s
#            
#            if not os.path.exists(copyTo): os.makedirs(copyTo)
#                
#            img = detectFace(cv2.imread(copyFrom))
#            print "Copy From:", copyFrom       
#            print "Copy To:", copyTo
#            print "-------------------"
#            copyTo = copyTo+"/"+str(fileNumber)+".png"
#            cv2.imwrite(copyTo, img)
#            fileNumber = fileNumber+1
            
            
     
            
#"""Extract the last frame of each sequence FACE"""  
#
#source =  "/data/datasets/Cohn-Kanade/Cohn-Kanade+/"      
#destination = "/data/datasets/Cohn-Kanade/ImagesCK_lastFrame_Face/"   
#
#fileNumber = 0
#for c in os.listdir(source):
#    for s in os.listdir(source+"/"+c):       
#        print "Folder:", source+"/"+c+"/"+s         
#        files = os.listdir(source+"/"+c+"/"+s)
#        if ".DS_Store" in files:
#            files.remove(".DS_Store")
#            
#        files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[2]))
#        copyFrom = source+"/"+c+"/"+s+"/"+files[-1]
#        copyTo = destination+"/"+c+"/"
#        
#        
#        if not os.path.exists(copyTo): os.makedirs(copyTo)
#            
#        img = detectFace(cv2.imread(copyFrom)) 
#        
#        copyTo = copyTo+"/"+str(fileNumber)+".png"
#        print "Copy From:", copyFrom       
#        print "Copy To:", copyTo
#        print "-------------------"
#        cv2.imwrite(copyTo, img)     
#        
#        fileNumber = fileNumber+1
#        
#        for a in range(1):
#            copyFromNeutral = source+"/"+c+"/"+s+"/"+files[a]
#            copyToNeutral = destination+"/Neutral"+"/"
#            if not os.path.exists(copyToNeutral): os.makedirs(copyToNeutral)
#            copyToNeutral = copyToNeutral+"/"+str(fileNumber)+".png"
#            
#            img = detectFace(cv2.imread(copyFromNeutral)) 
#            cv2.imwrite(copyToNeutral, img)  
#            #shutil.copyfile(copyFromNeutral,copyToNeutral)  
#            fileNumber = fileNumber+1
        
    

#"""Separate the files into feedback"""
#
#
#directory = "/export/experiments/emotionalFeedback/Cohn-Kanade+/"
#
#directorySave = "/export/experiments/emotionalFeedback/cohn-kanade+_Feedback_"
#
#for classes in sorted(os.listdir(directory)):    
#    
#    for sequence in sorted(os.listdir(directory+"/"+classes+"/")):             
#        
#        fileDir = os.listdir(directory+"/"+classes+"/"+sequence)
#        if ".DS_Store" in fileDir:
#            fileDir.remove(".DS_Store")
#        
#        imageDirectory = sorted(fileDir, key=lambda x: int(x.split('.')[0].split('_')[2]))
#        
#        imageDirectoryEmotion = imageDirectory[len(imageDirectory)-3 : len(imageDirectory)-1]        
#        imageDirectoryNeutral = imageDirectory[0:1] 
#        print "Neutral:", imageDirectoryNeutral
#        print "Emotion:", imageDirectoryEmotion
#        for image in  imageDirectoryNeutral:              
#            copyTo = directorySave+"/Neutral/"            
#            copyFrom = directory+"/"+classes+"/"+sequence+"/"+image                
#            if not os.path.exists(copyTo): os.makedirs(copyTo)
#                
#            #img = detectFace(cv2.imread(copyFrom))
#            #cv2.imwrite(copyTo+"/_"+classes+"_"+str(sequence)+"_"+image, img)
#            
#            shutil.copyfile(copyFrom,copyTo+"/_"+classes+"_"+str(sequence)+"_"+image)   
#
#
#        for image in  imageDirectoryEmotion:  
#            
#            if "Anger" in classes or "Disgust" in classes or "fear" in classes or "sadness" in classes or "Contempt" in classes:
#                copyTo = directorySave+"/Negative/"
#            else:
#                copyTo = directorySave+"/Positive/"
#                
#            copyFrom = directory+"/"+classes+"/"+sequence+"/"+image                
#            if not os.path.exists(copyTo): os.makedirs(copyTo)
#            
#            #img = detectFace(cv2.imread(copyFrom))
#            #cv2.imwrite(copyTo+"/_"+classes+"_"+str(sequence)+"_"+image, img)
#            shutil.copyfile(copyFrom,copyTo+"/_"+classes+"_"+str(sequence)+"_"+image)   
                
                
                                                           

#"""Separate the files in sequences of N frames"""
#
#frames = 4
#directory = "/export/experiments/emotionalFeedback/Cohn-Kanade+/"
#
#directorySave = "/export/experiments/emotionalFeedback/cohn-kanade+_Feedback_"+str(frames)+"/"
#currentSequenceNumber = 0
#for classes in sorted(os.listdir(directory)):    
#    
#    for sequence in sorted(os.listdir(directory+"/"+classes+"/")):             
#        currentSequenceImages = 0    
#        #imageDirectory = os.listdir(directory+"/"+subject+"/"+classes)       
#        fileDir = os.listdir(directory+"/"+classes+"/"+sequence)
#        if ".DS_Store" in fileDir:
#            fileDir.remove(".DS_Store")
#        
#        imageDirectory = sorted(fileDir, key=lambda x: int(x.split('.')[0].split('_')[2]))
#        imageDirectory = imageDirectory[len(imageDirectory)-6:len(imageDirectory)-1]
#        print "Directory:", directory+"/"+classes+"/"+sequence
#        print "imageDirectoryLen :", len(imageDirectory)
#        lastImage = ""
#        if len(imageDirectory)>0:
#            firstImage = True
#            for image in  imageDirectory:                    
#                if currentSequenceImages >= frames:
#                    currentSequenceNumber = currentSequenceNumber+1
#                    currentSequenceImages = 0                    
#                    
#                copyFrom = directory+"/"+classes+"/"+sequence+"/"+image
#                copyTo = directorySave+"/"+classes+"/"+str(currentSequenceNumber)+"/"
#                #print "copyto:", copyTo+"/"+image
#                if not os.path.exists(copyTo): os.makedirs(copyTo)
#                shutil.copyfile(copyFrom,copyTo+"/"+image)   
#                currentSequenceImages = currentSequenceImages+1
#                lastImage = copyFrom,copyTo+"/"
#            
#            
#            while currentSequenceImages < 4:           
#                shutil.copyfile(lastImage[0],lastImage[1]+str(currentSequenceImages)+".png")   
#                currentSequenceImages = currentSequenceImages+1
#                
#            currentSequenceNumber = currentSequenceNumber+1

#"""Separate the files in sequences of N frames"""
#
#frames = 4
#directory = "/export/experiments/emotionalFeedback/Cohn-Kanade+/"
#
#directorySave = "/export/experiments/emotionalFeedback/Cohn-Kanade/cohn-kanade/cohn-kanade+_Frames_"+str(frames)+"/"
#currentSequenceNumber = 0
#for subject in sorted(os.listdir(directory)):
#    
#    
#    for classes in sorted(os.listdir(directory+"/"+subject+"/")):             
#        currentSequenceImages = 0    
#        #imageDirectory = os.listdir(directory+"/"+subject+"/"+classes)        
#        imageDirectory = sorted(os.listdir(directory+"/"+subject+"/"+classes), key=lambda x: int(x.split('.')[0].split('_')[2]))
#        imageDirectory = imageDirectory[len(imageDirectory)-6:len(imageDirectory)-1]
#        print "Directory:", directory+"/"+subject+"/"+classes
#        print "imageDirectoryLen :", len(imageDirectory)
#        lastImage = ""
#        if len(imageDirectory)>0:
#            firstImage = True
#            for image in  imageDirectory:
#                    
#                if currentSequenceImages >= frames:
#                    currentSequenceNumber = currentSequenceNumber+1
#                    currentSequenceImages = 0                    
#                    
#                copyFrom = directory+"/"+subject+"/"+classes+"/"+image
#                copyTo = directorySave+"/"+classes+"/"+str(currentSequenceNumber)+"/"
#                #print "copyto:", copyTo+"/"+image
#                if not os.path.exists(copyTo): os.makedirs(copyTo)
#                shutil.copyfile(copyFrom,copyTo+"/"+image)   
#                currentSequenceImages = currentSequenceImages+1
#                lastImage = copyFrom,copyTo+"/"
#            
#            
#            while currentSequenceImages < 4:           
#                shutil.copyfile(lastImage[0],lastImage[1]+str(currentSequenceImages)+".png")   
#                currentSequenceImages = currentSequenceImages+1
#                
#            currentSequenceNumber = currentSequenceNumber+1
            
            


#"""Separate the files of the xtended CK"""
#
#frames = 1
#directory = "/export/cohn-kanade-images/"
#emotionDirectory = "/export/Emotion/"
#
#directorySave = "/export/experiments/emotionalFeedback/Cohn-Kanade+/"
#currentSequenceNumber = 0
#
#sequenceNumber = 0
#for subject in sorted(os.listdir(directory)):
#    for sequence in sorted(os.listdir(directory+"/"+subject+"/")):   
#        if not ".DS_Store" in sequence:
#            if os.path.exists(emotionDirectory + "/" + subject+"/"+sequence):
#                files = os.listdir(emotionDirectory + "/" + subject+"/"+sequence)   
#                if len(files)>0:
#                    f = open(emotionDirectory + "/" + subject+"/"+sequence+"/"+files[0], 'r')
#                    label = f.read().split(".")[0]                
#                    imageDirectory = sorted(os.listdir(directory+"/"+subject+"/"+sequence))
#                    
#                    for image in  imageDirectory:  
#                        copyFrom = directory+"/"+subject+"/"+sequence+"/"+image
#                        copyTo = directorySave+"/"+label+"/"+str(sequenceNumber)
#                        if not os.path.exists(copyTo): os.makedirs(copyTo)
#                        shutil.copyfile(copyFrom,copyTo+"/"+image) 
#                    sequenceNumber = sequenceNumber+1            
#              