# -*- coding: utf-8 -*-
import os
import shutil


sourceFolder = "/data/datasets/EMO_DB/wav/"

destinationFolder = "/data/datasets/EMO_DB/separated_wav"

fileNumber = 0
for f in os.listdir(sourceFolder):
    emotion = f[5]    
    sourceFile = sourceFolder+"/"+f
    destinyFile = destinationFolder+"/"+emotion+"/"
    if not os.path.exists(destinyFile):            
        os.makedirs(destinyFile)
        
    destinyFile = destinyFile+"/"+str(fileNumber)+".wav"
    shutil.copy(sourceFile, destinyFile)
    fileNumber = fileNumber+1
    
    
    
