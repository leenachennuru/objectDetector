# -*- coding: utf-8 -*-

import os
import shutil
import cv2
import numpy

    
        

#""" Copy the files from the matlab script output to a folder - Audio"""
#directory = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/data/avletters/Audio/mfcc/Clean/"
#
#directorySave1 = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/audio_GrayScale/"
#directorySave2 = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/audio_Raw/"
#
#for folder in os.listdir(directory):
#    if not ".mfcc" in folder and not ".DS_Store" in folder and not "convert.m" in folder:
#        files = os.listdir(directory+"/"+folder)
#        file1 = directory+"/"+folder+"/"+files[0]
#        file2 = directory+"/"+folder+"/"+files[1]
#         
#        label = folder[0:1]
#        
#                
#        img = cv2.imread(file1,0)
#        img = numpy.array(img).T
#        
##        rows,cols = img.shape
##        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
##        dst = cv2.warpAffine(img,M,(cols,rows))
#        
#                
#        newFolder = directorySave1 +"/" + label+"/"    
#        if not os.path.exists(newFolder): os.makedirs(newFolder)
#        
#        #print "copy:", file1, " ---to--- ", newFolder+"/"+files[0]
#        #shutil.copyfile(file1,newFolder+"/"+files[0]+".png")     
#        cv2.imwrite(newFolder+"/"+files[0]+".png", img)
#                
#        img = cv2.imread(file2,0)
#        img = numpy.array(img).T                
##        img = cv2.imread(file2,0)
##        rows,cols = img.shape
##        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
##        dst = cv2.warpAffine(img,M,(cols,rows))
##        
#        newFolder = directorySave2 +"/" + label+"/"    
#        if not os.path.exists(newFolder): os.makedirs(newFolder)
#        
#        #cv2.imwrite(newFolder+"/"+files[1]+".png", dst)
#        
#        #print "copy:", file1, " ---to--- ", newFolder+"/"+files[0]
#        shutil.copyfile(file2,newFolder+"/"+files[1]+".png")     
            
    
    


""" Copy the files from the matlab script output to a folder - Lips"""
directory = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/images/"
directorySave = "//informatik2/wtm/home/barros/Documents/Experiments/avLetters/images_sequence/"

#for sequence in os.listdir(directory):
#    
#    label = sequence[0:1]
#    newFolder = directorySave +"/" + label+"/" + sequence
#    #DataUtil.createFolder(newFolder)#
#    if not os.path.exists(newFolder): os.makedirs(newFolder)
#    
#    for image in os.listdir(directory+"/"+sequence+"/"):       
#        if not ".mat" in image:
#            #print "copy:", directory+"/"+sequence+"/"+image, " ---to--- ", newFolder+"/"+image+".png"
#            shutil.copyfile(directory+"/"+sequence+"/"+image,newFolder+"/"+image+".png")     
#    


#

 
"""Separate the files in sequences of N frames and synchronize with the audio """

frames = 4
directory = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/images_sequence/"
directoryAudio = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/audio_GrayScale/"

directorySave = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/images_sequence_"+str(frames)+"/"
directorySaveAudio = "/informatik2/wtm/home/barros/Documents/Experiments/avLetters/audio_synchronized_"+str(frames)+"/"

for classes in sorted(os.listdir(directory)):
    
    currentAudio = 0    
    currentSequenceNumber = 0
    for sequence in sorted(os.listdir(directory+"/"+classes+"/")):             
        currentSequenceImages = 0
        imageDirectory = sorted(os.listdir(directory+"/"+classes+"/"+sequence+"/"), key=lambda x: int(x.split('.')[0]))
        imageDirectory = imageDirectory[4:len(imageDirectory)-4]
        #print "Here:", imageDirectory
        lastImage = ""
        audioFiles = os.listdir(directoryAudio+"/"+classes)
        currentAudioFile = audioFiles[currentAudio]  
        
        firstImage = True
        for image in  imageDirectory:
         #   print image 
            if firstImage:
                copyFrom = directoryAudio+"/"+classes + "/"+currentAudioFile                
                copyTo = directorySaveAudio + "/" + classes + "/"         
                if not os.path.exists(copyTo): os.makedirs(copyTo)                    
                shutil.copyfile(copyFrom,copyTo+currentAudioFile+"_"+str(currentSequenceNumber)+".png")                           
                firstImage = False
                
            if currentSequenceImages >= frames:
                currentSequenceNumber = currentSequenceNumber+1
                currentSequenceImages = 0
                
                copyFrom = directoryAudio+"/"+classes + "/"+currentAudioFile                
                copyTo = directorySaveAudio + "/" + classes + "/"         
                if not os.path.exists(copyTo): os.makedirs(copyTo)                    
                shutil.copyfile(copyFrom,copyTo+currentAudioFile+"_"+str(currentSequenceNumber)+".png")                        
                

                
            copyFrom = directory+"/"+classes+"/"+sequence+"/"+image
            copyTo = directorySave+"/"+classes+"/"+str(currentSequenceNumber)+"/"
            #print "CopyFrom:", copyFrom
            #print "CopyTo:", copyFrom
            if not os.path.exists(copyTo): os.makedirs(copyTo)
            shutil.copyfile(copyFrom,copyTo+"/"+image)   
            currentSequenceImages = currentSequenceImages+1
            lastImage = copyFrom,copyTo+"/"
        
        
        while currentSequenceImages < 4:
            shutil.copyfile(lastImage[0],lastImage[1]+str(currentSequenceImages)+".png")   
            currentSequenceImages = currentSequenceImages+1
            
        currentAudio = currentAudio+1
        currentSequenceNumber = currentSequenceNumber+1
        
#        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    