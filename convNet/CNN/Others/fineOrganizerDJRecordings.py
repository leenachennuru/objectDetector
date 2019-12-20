# -*- coding: utf-8 -*-

import os
import shutil
import cv2
import numpy




"""Separate the files in sequences of N frames"""

frames = 4
directory = "/export/gestures/Subject1/"

directorySave = "/export/gestures/Subject1_Appex_1Frame/"
currentSequenceNumber = 0
for classes in sorted(os.listdir(directory+"/")):      
    for sequence in sorted(os.listdir(directory+"/"+"/"+classes+"/")): 
        currentSequenceImages = 0    
        imageDirectory = os.listdir(directory+"/"+"/"+classes+"/"+sequence+"/")        
        #imageDirectory = sorted(os.listdir(directory+"/"+subject+"/"+classes), key=lambda x: int(x.split('.')[0].split('_')[2]))
        imageDirectory = imageDirectory[40:50]
        print "Directory:", directory+"/"+"/"+classes
        
        for image in  imageDirectory:              
            copyTo = directorySave+"/"+classes+"/"         
            copyFrom = directory+"/"+"/"+classes+"/"+sequence+"/"+image                
            if not os.path.exists(copyTo): os.makedirs(copyTo)
            shutil.copyfile(copyFrom,copyTo+"/_"+classes+"_"+str(sequence)+"_"+image)   
        
                    