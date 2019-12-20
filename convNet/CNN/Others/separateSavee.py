# -*- coding: utf-8 -*-

import os
import shutil

source = "/data/datasets/SAVEE/AudioData//"
destination = "/data/datasets/SAVEE/AudioPerSubject/not_separated/"

i = 0

u = 0
for user in os.listdir(source):
    u = u +1
    print "user:", user
    #raw_input("here")
    amountPerSeparation = [0,0,0,0,0,0,0]
    for audio in os.listdir(source+"/"+user):
        print "Audio:", audio
        c = audio[0:-6]
        print "C:", c

        if c== "a":
            index = 0
        elif c=="d":
            index = 1
        elif c=="f":
            index = 2        
        elif c=="h":
            index = 3
        elif c=="n":
            index = 4
        elif c=="sa":
            index = 5    
        elif c=="su":
            index = 6    
          
                
#        if amountPerSeparation[index] < 8:
#            folder = "train"
#        elif amountPerSeparation[index] >= 8 and amountPerSeparation[index]<11:
#            folder = "validation"
#        else:
#            folder="test"
                    
#        if not os.path.exists(destination+"/"+user+"/"+folder+"/"+c+"/"):            
#                os.makedirs(destination+"/"+user+"/"+folder+"/"+c+"/")
#            
#        destinationAudio = destination+"/"+user+"/"+folder+"/"+c+"/"+str(u)+str(i)+".wav"
        if not os.path.exists(destination+"/"+user+"/"+c+"/"):            
             os.makedirs(destination+"/"+user+"/"+c+"/")
            
        destinationAudio = destination+"/"+user+"/"+c+"/"+str(u)+str(u)+str(u)+str(i)+".wav"
        
        shutil.copy(source+"/"+user+"/"+"/"+audio, destinationAudio)
        i = i+1
        amountPerSeparation[index] = amountPerSeparation[index]+1
    #print "amount per separation:", amountPerSeparation
                
    
    