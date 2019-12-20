# -*- coding: utf-8 -*-

import os

directory = "/data/datasets/AFEW/Validation_Audio/"

#Sequence renamer
#for c in os.listdir(directory):
#    fileNumber = 0
#    for s in os.listdir(directory+"/"+c):
#        for f in os.listdir(directory+"/"+c+"/"+s+"/"):
#            fileName = directory+"/"+c+"/"+s+"/"+f
#            fileNewName = directory+"/"+c+"/"+s+"/"+str(fileNumber)+".png"
#            os.rename(fileName, fileNewName)
#            fileNumber = fileNumber+1
        
#Static renamer
for c in os.listdir(directory):
    fileNumber = 0
    
    for f in os.listdir(directory+"/"+c):
        fileName = directory+"/"+c+"/"+f
        fileNewName = directory+"/"+c+"/"+str(fileNumber)+".wav"
        os.rename(fileName, fileNewName)
        fileNumber = fileNumber+1
        
