#!/usr/bin/env python

import shutil
import os
import math
import datetime
import cv2
import numpy
from PIL import Image, ImageChops


#directory = "/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/images/"
#directory2 = "/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/images2/"
#
#def applySkinSegmentation(image):
#    
#    im_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
#    skin_ycrcb_mint = numpy.array((0, 133, 77))
#    skin_ycrcb_maxt = numpy.array((255, 173, 127))
#    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)       
#    
#    for x in range(len(skin_ycrcb)):
#        for y in range(len(skin_ycrcb[x])):            
#            if skin_ycrcb[x][y]==0:            
#                skin_ycrcb[x][y] = 255
#            else:
#                skin_ycrcb[x][y] = 0    
#                
#    #cv2.imwrite(directory, skin_ycrcb)
#    return skin_ycrcb
#        
#
#for folder in os.listdir(directory):
#    for image in os.listdir(directory+"/"+folder):
#        img =  Image.open(directory+"/"+folder+"/"+image)     
#        w, h = img.size
#
#        img = img.crop((0,0,640,170))
#        #img.save(directory2+"/"+folder+"/"+image)
#        img = numpy.array(img)
#        b,g,r = cv2.split(img)
#        img = cv2.merge([r,g,b])
#        cv2.imwrite("/informatik/isr/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/images/test.jpg", img)
#        break
#    
        #cv2.imshow('image',numpy.array(img))
        #cv2.waitKey(20)
        #break
    #break
#        img = cv2.imread(directory+"/"+folder+"/"+image)
#        img = applySkinSegmentation(img)
#        cv2.imwrite(directory2+"/"+folder+"/"+image,img)
        


#for folder in os.listdir(directory):
#    images = directory + "/" + folder
#    for i in os.listdir(images):
#        print i
#        fileName = str(datetime.datetime.now())+".jpg"
#        print directory+"/"+folder+"/"+i
#        #os.rename(directory+"/"+folder+"/"+i, directory+"/"+folder+"/"+fileName)
#        
#        if "left" in folder:
#            #os.rename(directory+"/"+"left"+"/"+fileName, directory+"/"+folder+"/"+fileName)
#            shutil.copyfile(directory+"/"+folder+"/"+i,directory2+"/"+"left"+"/"+i)     
#            
#        elif "right" in folder:
#          #  os.rename(directory+"/"+"right"+"/"+i, directory+"/"+folder+"/"+fileName)    
#            shutil.copyfile(directory+"/"+folder+"/"+i,directory2+"/"+"right"+"/"+i)     
#                 
#        elif "middle" in folder:
##            os.rename(directory+"/"+"right"+"/"+i, directory+"/"+folder+"/"+fileName)  
#            shutil.copyfile(directory+"/"+folder+"/"+i,directory2+"/"+"middle"+"/"+i)     

#directory = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/Body_temporal_label_segmentation/"
#saveDirectory = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/body_sequences_full_2"
#
#for c in os.listdir(directory):
#    sequence  = 0        
#    for s in os.listdir(directory+"/"+c+"/"):        
#        images = []
#        for i in os.listdir(directory+"/"+c+"/"+s+"/"):                        
#            images.append(i)            
#            if len(images)==2:                
#                sequence = sequence+1
#                for h in images:
#                    if not os.path.exists(saveDirectory+"/"+c+"/"+str(sequence)): os.makedirs(saveDirectory+"/"+c+"/"+str(sequence))
#                    shutil.copyfile(directory+"/"+c+"/"+"/"+s+"/"+h, saveDirectory+"/"+c+"/"+str(sequence)+"/"+h)
#                images = []
#        
        



"""
directory = "/informatik2/wtm/home/barros/Documents/Experiments/handPostureDataset/images/RGB/testing/"

saveIn = "/informatik2/wtm/home/barros/Documents/Experiments/handPostureDataset/images/testing/"

persons = os.listdir(directory)
fileNames = [0,0,0,0,0,0,0,0,0,0]

for k in persons:
    backgrounds = os.listdir(directory+"/"+k)
    for b in backgrounds:
        illuminations = os.listdir(directory+"/"+k+"/"+b)
        for i in illuminations:
            poses = os.listdir(directory+"/"+k+"/"+b+"/"+i)
            for p in poses:
                actions = os.listdir(directory+"/"+k+"/"+b+"/"+i+"/"+p)
                aNumber = 0
                for a in actions:
                    files = os.listdir(directory+"/"+k+"/"+b+"/"+i+"/"+p+"/"+a)
                    fileNames[aNumber] = fileNames[aNumber]+1
                    for f in files:                
                        save = saveIn+"/"+str(a)+"/"+str(fileNames[aNumber])+"/"
                        
                        if not os.path.exists(save): os.makedirs(save)
                        
                        save = saveIn+"/"+str(a)+"/"+str(fileNames[aNumber])+"/"+f
                        
                        print directory+"/"+k+"/"+b+"/"+i+"/"+p+"/"+a+"/"+f
                        shutil.copyfile(directory+"/"+k+"/"+b+"/"+i+"/"+p+"/"+a+"/"+f, save)
                        
                        #print save
                
                    aNumber = aNumber+1    
                
                
        
   """     




#imageDirectory = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/Body_temporal_label_segmentation/"
#saveDirectory = "/informatik/isr/wtm/home/barros/Documents/Experiments/FABO/Body_static_temporal_label/"
#classesPath = os.listdir(imageDirectory)
#examples = {"HAPPINESS": 0, "PST SRP":0, "NGT SRP": 0, "BOREDOM":0, "DISGUST": 0, "FEAR":0, "UNCERTAINTY": 0, "PUZZLEMENT":0, "SADNESS": 0, "ANGER":0, "ANXIETY":0, "NEUTRAL":0  }
#
#for c in classesPath:
#    directory = imageDirectory +"/" + c
#    sequences = os.listdir(directory)
#    for s in sequences:
#        sequenceDirectory = directory + "/"+s    
#
#        images = os.listdir(sequenceDirectory)
#        for i in images:
#            directoryFrom = sequenceDirectory + "/" + i
#            directoryTo = saveDirectory +"/"+c            
#            print "-"
#            if not os.path.exists(directoryTo): os.makedirs(directoryTo)
#            directoryTo = directoryTo + "/"+str(examples[c])+i
#            print directoryFrom
#            print directoryTo
#            shutil.copyfile(directoryFrom, directoryTo)
#            examples[c] = examples[c]+1
    


"""
faceDirectory = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/homes/hatice/FABO_released/FABO_DVD1/face_camera_part2/"
classesPath = os.listdir(faceDirectory)

for s in classesPath:
    directory = faceDirectory +"/" + s+"/face_body"
    videos = os.listdir(directory)
    for f in videos:
        directoryFrom = directory + "/"+f
        
        directoryTo = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/faceVideos"+"/"+s
        print directoryFrom
        print directoryTo
        print "-"
        #if not os.path.exists(directoryTo): os.makedirs(directoryTo)
        #directoryTo = directoryTo + "/"+f
        #shutil.copyfile(directoryFrom, directoryTo)
        

"""

# GETING FRAMES FROM ONSET UNTIL OFFSET
csvDirectoryTemporal = "/informatik/isr//wtm/home/barros/Documents/Experiments/FABO/body_temporal_separation.csv"
csvDirectoryCategory = "/informatik/isr//wtm/home/barros/Documents/Experiments/FABO/body_Label_separation.csv"
copyFromDirectory = "/informatik/isr//wtm/home/barros/Documents/Experiments/FABO/Body/"
copyToDirectory = "/informatik/isr//wtm/home/barros/Documents/Experiments/FABO/Body_temporal_label_segmentation/"

fTemporal = open(csvDirectoryTemporal, 'r')  
fCategory = open(csvDirectoryCategory, 'r')  

examples = {"HAPPINESS": 0, "PST SRP":0, "NGT SRP": 0, "BOREDOM":0, "DISGUST": 0, "FEAR":0, "UNCERTAINTY": 0, "PUZZLEMENT":0, "SADNESS": 0, "ANGER":0, "ANXIETY":0, "NEUTRAL":0  }

x = 0
for lineTemporal in fTemporal:    
    lineCategory = fCategory.readline()

    print "LineTemporal : " , lineTemporal    
    print "LineCategory : " , lineCategory    
    
    li = lineTemporal.split(",")
    liCategory = lineCategory.split(",")
    folderName = li[0].split("-")[0]
    fileNumber = li[0].split("-")[1]
       
    directoryCopyFrom = copyFromDirectory + "/"
    category = liCategory[1]
    print  "------ Category: ", category

    if category == "HAPPINESS" or category == "PST SRP" or category == "NGT SRP" or category == "BOREDOM" or category == "DISGUST" or category == "FEAR" or category == "UNCERTAINTY" or category == "PUZZLEMENT" or category == "ANXIETY" or category == "SADNESS" or category == "ANGER":
        
        
        
        # FIRST SEQUENCE ON THE VIDEO, STARTING WITH NEUTRAL, GETING THE GESTURE, FINISHING WITH NEUTRAL.
        
        start = li[1]
        end = li[2]    
        print  "------ Start Neutral:", start
        print  "------ End Neutral:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start)+1, int(end)):                        
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
                
        start = li[2]
        end = li[7]
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples[category] = examples[category]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + category + "/" + str(examples[category]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/"): os.makedirs(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
        
        start = li[7]
        end = li[8]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                "Neutral"
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass        
            
        
        
        # SECOND SEQUENCE ON THE VIDEO, STARTING WITH NEUTRAL, GETING THE GESTURE, FINISHING WITH NEUTRAL.
        
        start = li[10]
        end = li[11]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                    
        
        start = li[11]
        end = li[16]
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples[category] = examples[category]+1        
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + category + "/" + str(examples[category]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/"): os.makedirs(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
        
        start = li[16]
        end = li[17]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass   
    
        # THIRD SEQUENCE ON THE VIDEO, STARTING WITH NEUTRAL, GETING THE GESTURE, FINISHING WITH NEUTRAL.
        
        start = li[18]
        end = li[19]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
                
        start = li[19]
        end = li[22]
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples[category] = examples[category]+1
            for h in xrange(int(start), int(end)):        
               
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + category + "/" + str(examples[category]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/"): os.makedirs(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
        
        start = li[22]
        end = li[23]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass 

        # FOURTH SEQUENCE ON THE VIDEO, STARTING WITH NEUTRAL, GETING THE GESTURE, FINISHING WITH NEUTRAL.
        
        start = li[24]
        end = li[25]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
        
        start = li[25]
        end = li[28]
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples[category] = examples[category]+1        
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + category + "/" + str(examples[category]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/"): os.makedirs(copyToDirectory + "/" + category + "/" + str(examples[category]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
                
        
        start = li[28]
        end = li[29]    
        print  "------ Start:", start
        print  "------ End:", end
        try:
            examples["NEUTRAL"] = examples["NEUTRAL"]+1
            for h in xrange(int(start), int(end)):        
                
                    directoryFrom = copyFromDirectory + "/" + folderName + "/" + fileNumber+".avi/"+str(h)+".jpg"
                    directoryTo = copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/" + str(h)+".jpg"
                    #directoryTo = copyToDirectory + "/" + category + "/" + "/" + str(x)+".jpg"
                    print "------------ From: ", directoryFrom
                    print "------------ To: ",  directoryTo
                    if not os.path.exists(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/"): os.makedirs(copyToDirectory + "/" + "Neutral" + "/" + str(examples["NEUTRAL"]) + "/")
                    #if not os.path.exists(copyToDirectory + "/" + category + "/" ): os.makedirs(copyToDirectory + "/" + category + "/" )                
                    shutil.copyfile(directoryFrom, directoryTo)
                    x = x+1
        except:
                pass
#    x = x+1
