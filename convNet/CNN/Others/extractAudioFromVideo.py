# -*- coding: utf-8 -*-

import os

import subprocess
from pydub import AudioSegment

sourceFolder = "/data/datasets/AFEW/raw_data/Train/videos/"
destinationFolder = "/data/datasets/AFEW/Train_Audio_MP3/"
destinationFolderWAV = "/data/datasets/AFEW/Train_Audio_WAV/"

#
#sound = AudioSegment.from_mp3("a.mp3")
#sound.export("/data/datasets/AFEW/Train_Audio_MP3/Neutral/file.wav", format="wav")

raw_input("here")

for c in os.listdir(sourceFolder):
    for videoFile in os.listdir(sourceFolder+"/"+c):
        videoSource = sourceFolder+"/"+c+"/"+videoFile
        audioDestination = destinationFolder+"/"+c+"/"
        audioDestinationWAV = destinationFolderWAV+"/"+c+"/"
        if not os.path.exists(audioDestination):            
            os.makedirs(audioDestination)
        #print "Video Source:", videoSource
        #raw_input("here")
        audioDestination = audioDestination+videoFile.split(".")[0]    
#        command = "mplayer -dumpaudio -dumpfile "+audioDestination+".mp3 "+videoSource
        command = "avconv -i "+videoSource+" -vn -f wav "+audioDestination 
        subprocess.call(command, shell=True)        
        
        
        #avconv -i /data/datasets/AFEW/raw_data/Train/videos/Neutral/003044960.avi -vn -f wav /data/test.wav

#        sound = AudioSegment.from_mp3(audioDestination+".mp3 ")
#        sound.export(audioDestinationWAV, format="wav")
        
        
            



