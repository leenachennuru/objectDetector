# -*- coding: utf-8 -*-

from Utils import DataUtil
from Utils import AudioProcessingUtil
import MCCNNExperiments
from Networks import MCCNN
import os
import numpy
import cv2

from sklearn.metrics import classification_report

def loadExperiment():
    
    
    modelDirectory = "/data/datasets/SAVEE//experiments/MFCC_512//model//repetition_0_BestTest_MFCC_512_.save"
    #/data/datasets/SAVEE//experiments/MFCC_512//model//repetition_0_BestTest_MFCC_512_.save
    #/data/datasets/SAVEE//experiments/MFCC_512//model//repetition_0_BestValidation_MFCC_512_.save
    #/data/datasets/SAVEE//experiments/MFCC_512//model//repetition_0_FINAL_MFCC_512_.save
    
    networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectory)   
    
    print networkState
            
    saveNetworkParameters = [False]
    
    network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )
    
    
    testingAudio = "/data/datasets/SAVEE/AudioPerSubject/DC/test"
    #testingAudio = "/data/datasets/SAVEE/AudioPerSubject/DC_JE/test"
    sequenceSizeInSeconds = 1
    
    trueData = []
    predictedData = []
    
    for c in os.listdir(testingAudio):
        for f in os.listdir(testingAudio+"/"+c):
            audioFile = testingAudio+"/"+c+"/"+f            
            print audioFile
            #audioMFCCs = AudioProcessingUtil.extractMFCC(audioFile, 0, 10) 
            audioMFCCs = AudioProcessingUtil.extractMFCC(audioFile, "", 0, sequenceSizeInSeconds)
            
            predictions = []
            for audioMFCC in audioMFCCs:
                cv2.imwrite("/data/datasets/mini-genres/test.png", audioMFCC)
                audioMFCC = cv2.imread("/data/datasets/mini-genres/test.png")
                #audioMFCC = audioMFCC.reshape(1, 13,21)
                #print "Shape:", numpy.array(audioMFCC).shape                
                result = MCCNN.classify(network[len(network)-2],[audioMFCC],trainingParameters[4])[0]*100                
                predicted = numpy.argmax(numpy.array(result), axis=0)
                predictions.append(predicted)      
                
            if not predictions == []:                
                finalPrediction = numpy.bincount(predictions).argmax()
                if c == "a":
                    trueData.append(0)
                elif c == "d":
                    trueData.append(1)
                elif c == "f":
                    trueData.append(2)
                elif c == "h":
                    trueData.append(3)
                elif c == "n":
                    trueData.append(4)    
                elif c == "sa":
                    trueData.append(5)                  
                elif c == "su":
                    trueData.append(6)                  
                predictedData.append(finalPrediction)    
                print "Audio class: ", c
                print "Audio file: ", audioFile
                print "Prediction:", predictions
                print "Final Vote:", finalPrediction
                print"-----------"
            
            
    print classification_report(trueData,predictedData, target_names=["a", "d", "f", "h", "n", "sa", "su"])   
    
    
    


loadExperiment()