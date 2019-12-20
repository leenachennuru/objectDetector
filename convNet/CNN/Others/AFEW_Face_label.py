# -*- coding: utf-8 -*-
import os
import shutil

#copy the aligned faces to a structured directory
rawFaceFolder = "/data/datasets/AFEW/raw_data/Train/Train_Aligned_Faces/"

videoNamesFolder = "/data/datasets/AFEW/raw_data/Train/videos/"

saveIn = "/data/datasets/AFEW/Train_Faces_Only"


for c in os.listdir(videoNamesFolder):
    for video in os.listdir(videoNamesFolder+"/"+c+"/"):        
        folderName = video.split(".")[0]        
        sourceFolder = rawFaceFolder+"/"+folderName
        
        destinyFolder = saveIn+"/"+c+"/"
        if not os.path.exists(destinyFolder):            
            os.makedirs(destinyFolder)
        print "Source:", sourceFolder
        print "Destiny:", destinyFolder
        print "------------------------------"
        try:
            shutil.copytree(sourceFolder, destinyFolder+"/"+folderName)
        except:
            print "Folder not found:", sourceFolder
    

#copy each folder to a four 