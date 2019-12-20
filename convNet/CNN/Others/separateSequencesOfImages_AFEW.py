# -*- coding: utf-8 -*-
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import scipy.io.wavfile
import numpy
import pylab

import cv2

def createSpectrumImage(wavDirectory, saveDirectory):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
    data = numpy.swapaxes(data,0,1)    
    
    #pylab.figure(frameon=False)    
    fig, (ax1) = pylab.subplots(nrows=1)
    spectrum, freqs, bins, im = ax1.specgram(data[0]) 
    
    ax1.set_axis_off()
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    
    fig.savefig(saveDirectory, bbox_inches='tight', pad_inches = 0)
    pylab.close()
    
    
    
    
FramesPerDirectory = 4

sourceDirectory = "/data/datasets/AFEW/Train_Faces_Only"

saveDirectory = "/data/datasets/AFEW/Train_Faces_Only_"+str(FramesPerDirectory)+"_Frames"


audioDirectory = "/data/datasets/AFEW/Train_Audio/"
saveAudioDirectory = "/data/datasets/AFEW/Train_Audio_Sychronized_"+str(FramesPerDirectory)+"_Frames/"

for c in os.listdir(sourceDirectory):
    if not c =="Neutral" or not c=="Sad" or not c=="Surprise":
        newSequenceDirectoryNumber = 0
        for sequence in os.listdir(sourceDirectory+"/"+c+"/"):
            files = os.listdir(sourceDirectory+"/"+c+"/"+sequence+"/")        
    
            files.sort()
    
            numberOfNewSequences = len(files) / FramesPerDirectory
            restOfSequences = len(files) % FramesPerDirectory
                
            print sourceDirectory+"/"+c+"/"+sequence+"/"
            print "Number of newSequences: ", numberOfNewSequences
            fileIndex = 0        
            for newSequenceIndex in range(numberOfNewSequences):
                newSequenceDirectory = saveDirectory+"/"+c+"/"+str(newSequenceDirectoryNumber)+"/"
                if not os.path.exists(newSequenceDirectory):            
                    os.makedirs(newSequenceDirectory)
                    
                if not os.path.exists(saveAudioDirectory+"/"+c+"/"):            
                    os.makedirs(saveAudioDirectory+"/"+c+"/")
                print "Reading audio:", audioDirectory+"/"+c+"/"+sequence.split(".")[0]   
                audioSpectogram = createSpectrumImage(audioDirectory+"/"+c+"/"+sequence.split(".")[0], saveAudioDirectory+"/"+c+"/"+str(newSequenceDirectoryNumber)+".png")            
                
                newSequenceDirectoryNumber = newSequenceDirectoryNumber+1
    
        
                for i in range(FramesPerDirectory):            
                    sourceFile = sourceDirectory+"/"+c+"/"+sequence+"/"+files[fileIndex]
                    
                    print "---File to be copied:", sourceFile
                    
                    shutil.copy(sourceFile, newSequenceDirectory+"/"+str(i)+".jpg")
                    print "Copy to:", newSequenceDirectory+"/"+str(i)+".png"                
                    fileIndex = fileIndex+1
                
                
                
                print "Directory:", newSequenceDirectory
                print "Audio_Directory:", saveAudioDirectory+"/"+c+"/"+sequence.split(".")[0]+"_"+str(newSequenceDirectoryNumber)+".png"    
            
            
            newSequenceDirectory = saveDirectory+"/"+c+"/"+str(newSequenceDirectoryNumber)+"/"
            if not os.path.exists(newSequenceDirectory):            
                    os.makedirs(newSequenceDirectory)
                    
            for restSequenceIndex in range (restOfSequences):
                    sourceFile = sourceDirectory+"/"+c+"/"+sequence+"/"+files[fileIndex]                
    
                    print "---File to be copied:", sourceFile
                    
                    shutil.copy(sourceFile, newSequenceDirectory+"/"+str(restSequenceIndex)+".jpg")
                    fileIndex = fileIndex+1
                    
            fileIndex = fileIndex-1
            destinyIndex = 0
            for restSequenceIndex in range(FramesPerDirectory-restOfSequences):
                sourceFile = sourceDirectory+"/"+c+"/"+sequence+"/"+files[fileIndex]
                
               
                destinyFile = newSequenceDirectory+"/"+ str(FramesPerDirectory - restSequenceIndex)+".jpg"
                print "---File to be copied:", sourceFile
                shutil.copy(sourceFile, destinyFile)
                destinyIndex = destinyIndex+1
                #raw_input("heree")     
            
            print "Directory:", newSequenceDirectory    
            audioSpectogram = createSpectrumImage(audioDirectory+"/"+c+"/"+sequence.split(".")[0], saveAudioDirectory+"/"+c+"/"+str(newSequenceDirectoryNumber)+".png")
            newSequenceDirectoryNumber = newSequenceDirectoryNumber+1                
            print "Audio_Directory:", audioDirectory+"/"+c+"/"+sequence.split(".")[0]
            #raw_input("here")
        
                
        
        


