# -*- coding: utf-8 -*-

import os

import subprocess

sourceFolder = "/informatik2/wtm/home/barros/Documents/Presentations/humanoids_2014/"
destinationFolder = "/informatik2/wtm/home/barros/Documents/Presentations/humanoids_2014_wav"


for c in os.listdir(sourceFolder):
    audioFileNumber = 0
    for audioFile in os.listdir(sourceFolder+"/"+c):
        audioSource = sourceFolder+"/"+c+"/"+audioFile
        audioDestination = destinationFolder+"/"+c+"/"
        
        if not os.path.exists(audioDestination):            
            os.makedirs(audioDestination)
        #print "Video Source:", videoSource
        #raw_input("here")
        audioDestination = audioDestination+str(audioFileNumber)+".wav"    
#        command = "mplayer -dumpaudio -dumpfile "+audioDestination+".mp3 "+videoSource
        #mplayer -ao pcm:file=targetfile.wav sourcefile.m4a
        print "AudioSource:", audioSource
        print "AudioDestination:", audioDestination
        print "--------------------------------"
        command = "mplayer -ao pcm:file="+audioDestination+" "+audioSource 
        subprocess.call(command, shell=True)        
        audioFileNumber = audioFileNumber+1