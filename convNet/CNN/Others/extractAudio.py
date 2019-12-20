# -*- coding: utf-8 -*-

import os
import numpy
import cv2

img = cv2.imread("/data/datasets/mini-genres/mini-genres_original_size_Butterworth_filter/classical/0.png")

print "Shape:", numpy.array(img).shape


import Image

img = "/data/datasets/mini-genres/mini-genres_original_size_Butterworth_filter/classical/0.png"

img = Image.open(img ).convert("RGB")
img.save("/data/005.bmp","bmp")

#I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
#
#img = Image.fromarray(I8)
#img.save("file.png")

#import subprocess
#
#sourceFolder = "/data/datasets/mini-genres/audio_wav/"
#destinationFolder = "/data/datasets/mini-genres/arss_spectrum/"
#
#
#for c in os.listdir(sourceFolder):
#    audioFileNumber = 0
#    for audioFile in os.listdir(sourceFolder+"/"+c):
#        audioSource = sourceFolder+"/"+c+"/"+audioFile
#        audioDestination = destinationFolder+"/"+c+"/"
#        
#        if not os.path.exists(audioDestination):            
#            os.makedirs(audioDestination)
#        #print "Video Source:", videoSource
#        #raw_input("here")
#        audioDestination = audioDestination+str(audioFileNumber)+".png"    
##        command = "mplayer -dumpaudio -dumpfile "+audioDestination+".mp3 "+videoSource
#        #mplayer -ao pcm:file=targetfile.wav sourcefile.m4a
#        print "AudioSource:", audioSource
#        print "AudioDestination:", audioDestination
#        print "--------------------------------"
#        command = "/data/arss "+audioSource+" "+audioDestination+ " --min-freq 20 --max-freq 20000 --bpo 12 --pps 10"
#
#        subprocess.call(command, shell=True)        
#        audioFileNumber = audioFileNumber+1