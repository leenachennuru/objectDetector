# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
    
from Utils import DataUtil

from Networks import MCCNN

import MCCNNExperiments
import os
import numpy

import cv2

def resize(image, size):            
        return numpy.array(cv2.resize(image,size))
        
def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        #Haarcascade file for face detection
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(img, 1.2, 4, 1, (20,20))
    
        if len(rects) == 0:            
            return None
        rects[:, 2:] += rects[:, :2]
        
        return rects
        return box(rects,img2, img2)

def box(rects, img, img2):        
        imgs = []
        for x1, y1, x2, y2 in rects:            
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 255, 0), 2)            
            imgs.append(img[y1:y2, x1:x2])
                        
        return imgs, img2

#Directory of the saved network
modelDirectory =  "EmotionRecognitionNetwork.save"
    
networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectory)       

experimentParameters[0] = os.path.dirname(os.path.abspath(__file__))
experimentParameters.append(False)
    
saveNetworkParameters = [False]

#Loading the network and creating the theano model
network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )    
    
    
#Open the webcam
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0) 
vc.set(3,640)
vc.set(4,480)
if vc.isOpened(): # try to get the first frame
    rval, f = vc.read()
 #   rval2, f2 = vc2.read()
else:
    rval = False
frameNumber = 0    
faces = []
lookTo = 0, 0
recognized = False


while rval:    
    frameNumber = frameNumber + 1
    #Every 3 frames, look for a face
    if frameNumber == 3:
        frameNumber = 0      
        
        #Look for a face
        rects = detectFace(f)   
                                                                
        #If the face is detected
        if not rects == None:           
            #Crop the face (Only working with one face per time - the last found face on the image)
            for x1, y1, x2, y2 in rects:                        
                img = f[y1:y2, x1:x2]       
                
            #Prepare the image to be send to the network    
            img = resize(img,networkTopology[6][0][0][4])    
            faces.append(img)
            
            #Once you collected 4 faces, classify them
            if len(faces) == 4:
                
                #Classify the face, returns the softmax probabilities 
                result = MCCNN.classify(network[len(network)-2],[[faces[0],faces[1],faces[2],faces[3]],faces[2]],trainingParameters[4])[0]*100                
                faces = []                
                
                predicted = numpy.argmax(numpy.array(result), axis=0)        
                
                #Center of the last detected face
                lookTo = ((x2 + x1)/2), ((y2+y1)/2)  
                
                if predicted == 0:
                        color = (192, 19, 19)
                        text = "Negative"                                                        
    
                         
                elif predicted == 2:
                        color = (19, 71, 192)                                                               
                        text = "Positive"                    
    
                else:

                        color = (200, 200, 20) 
                        text = "Neutral"       
                recognized = True
    
    
        else:
            recognized = False
    
    #Once there is a recognition, paint a square on the image with the face & recognition result.
    if recognized:        
        
        cv2.circle(f, lookTo, 10, color, thickness=-1)                                                                   
        cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)                
        cv2.putText(f,text, (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color)                                                        
        
    cv2.imshow("FaceLiveImage", f)                 
    rval, f = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break       
         