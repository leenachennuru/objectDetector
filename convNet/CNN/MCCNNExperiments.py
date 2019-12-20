# -*- coding: utf-8 -*-


from Networks import MCCNN
from Utils import LogUtil
from Utils import DataUtil
from Utils import ImageProcessingUtil
from Utils import AudioProcessingUtil

import os
import datetime
import numpy
import cv2


import Image


def runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters ):
        
    

    channels =  networkTopology[6]
    outputUnits = networkTopology[5][0]
    loadImagesStrategy = networkTopology[7]
    preLoadedFilters = networkTopology[8]
    
    batchSize = trainingParameters[4]                
            
    baseDirectory = experimentParameters[0]
    imagesDirectory = experimentParameters[1]
    experimentName =  experimentParameters[2]
    isGeneratingMetrics = experimentParameters[4]    
    
    isSavingNetwork = saveNetworkParameters[0]
    
    
    isTraining = trainingParameters[0]
    
    imagesDirectory = baseDirectory + imagesDirectory 
        
    experimentDirectory = baseDirectory + "/experiments/"+experimentName+"/"
    
    metricsDirectory = experimentDirectory+"/metrics/"    
    metricsFile = metricsDirectory+"/Log/Metrics_"+experimentName+".txt"
        
    trainingEpochesDirectory = metricsDirectory+"/trainingEpoches/"

    
    convFeaturesDirectory=experimentDirectory+"/convFeatures/"
    modelDirectory=experimentDirectory+"/model/"    
    hintonDiagram=experimentDirectory+"/hintonDiagram/"
    outputImages=experimentDirectory+"/outputImages/"
    saveHistoryImageFiltersDirectory = experimentDirectory+"/"+"Filters"
    
    
    log = LogUtil.LogUtil()
    log.createLog(experimentName,metricsDirectory)
    
    log.createFolder(metricsDirectory)
    log.createFolder(trainingEpochesDirectory)
    log.createFolder(convFeaturesDirectory)
    log.createFolder(modelDirectory)
    log.createFolder(hintonDiagram)          
    log.createFolder(saveHistoryImageFiltersDirectory)
    log.createFolder(outputImages)     
    
    
    precisionSynchronized = []
    recallSynchronized = []
    fScoreSynchronized = []
    accuracySynchronized = []
    
    precisionSynchronizedPerClass = []
    recallSynchronizedPerClass = []
    fScoreSynchronizedPerClass = []
    
    precisionMicro = []
    precisionMacro = []
    precisionWeighted = []
    
    recallMicro = [] 
    recallMacro = []
    recallWeighted = []
        
    fScoreMicro = []
    fScoreMacro = []
    fScoreWeighted = []

    accuracy = []

    precisionsPerClass = []
    recallsPerClass = []
    fScoresPerClass = [] 
    
    
    classesInEachSet = []    
    
    trainingTime = []
    recognitionTime = []                
                                         
    
    log.startNewStep("Network Parameters")
    log.startNewStep("Network Topology")
    log.printMessage(("Load images strategy: ", networkTopology[7]))    
    log.startNewStep("Channels Topology")
    channelNumber = 0
    for channel in networkTopology[6]:
        log.printMessage(("Channel: ", channelNumber))        
        log.printMessage(("Connect to hidden layer: ", channel[2]))
        
        log.printMessage(("Input Modality: ", channel[0][1]))
        log.printMessage(("---Input images: ", channel[0][0]))        
        log.printMessage(("---Image size: ", channel[0][4]))      
        log.printMessage(("---Image structure: ", channel[0][2]))
        log.printMessage(("---Feature type: ", channel[0][6]))
        log.printMessage(("---Data directory "))      
            
        if len(channel[0][3]) == 1:
            log.printMessage(("------ Folder: ", channel[0][3][0][0]))                    
            log.printMessage(("------ Percent Training: ", channel[0][3][0][1]))
            log.printMessage(("------ Percent Validation: ", channel[0][3][0][2]))
            log.printMessage(("------ Percent Testing: ", channel[0][3][0][3]))
        elif len(channel[0][3]) == 2:
            log.printMessage(("------ Folder: ", channel[0][3][0][0]))                    
            log.printMessage(("------ N-Fold Cross validation sets: ", channel[0][3][0][1]))
        else:
            log.printMessage(("------ Folder Training: ", channel[0][3][0]))                    
            log.printMessage(("------ Percent Validation: ", channel[0][3][1]))
            log.printMessage(("------ Percent Testing: ", channel[0][3][2]))
            
        log.printMessage(("---Data augmentation: ", channel[0][5]))
        log.printMessage(("---Image Position: ", channel[0][7]))
    
        channelNumber = channelNumber+1        
        layerNumber = 0
        for layer in channel[1]:
            log.printMessage((" --- Layer: ", layerNumber))
            layerNumber = layerNumber+1
            log.printMessage((" ------ Filters: ", layer[0]))            
            log.printMessage((" ------ Filter dimension: ", layer[1]))
            log.printMessage((" ------ Max-Pooling: ", layer[2], "(", layer[3],")"))            
            log.printMessage((" ------ Shunting inhibition: ", not(layer[4]==None)))
            if layer[4] != None:
                log.printMessage((" --------- Shunting inhibition type: ", layer[4][0]))                                
                log.printMessage((" --------- Trainable: ", layer[4][1]))
            log.printMessage((" ------ Layer type: ", layer[5]))
            log.printMessage((" ------ Activation Function: ", layer[6]))
            log.printMessage((" ------ L1 Regularization: ", layer[7]))
            log.printMessage((" ------ L2 Regularization:  ", layer[8]))
            log.printMessage((" ------ Trainable: ", layer[9]))
            
    if not networkTopology[9]  == []:
        log.startNewStep("Cross Channel Layers")
        channelNumber = 0
        for channel in networkTopology[9]:
            log.printMessage(("CrossChannel: ", channelNumber))
            for inputChannel in channel[0]:
                log.printMessage(("---Input channel: ", inputChannel[0]))
                log.printMessage(("---Input channel layer: ", inputChannel[1]))
                       
            channelNumber = channelNumber+1        
            layerNumber = 0
            for layer in channel[1]:
                log.printMessage((" --- Layer: ", layerNumber))
                layerNumber = layerNumber+1
                log.printMessage((" ------ Filters: ", layer[0]))            
                log.printMessage((" ------ Filter dimension: ", layer[1]))
                log.printMessage((" ------ Max-Pooling: ", layer[2], "(", layer[3],")"))            
                log.printMessage((" ------ Shunting inhibition: ", not(layer[4]==None)))
                if layer[4] != None:
                    log.printMessage((" --------- Shunting inhibition type: ", layer[4][0]))                                
                    log.printMessage((" --------- Trainable: ", layer[4][1]))
                log.printMessage((" ------ Layer type: ", layer[5]))
                log.printMessage((" ------ Activation Function: ", layer[6]))
                log.printMessage((" ------ L1 Regularization: ", layer[7]))
                log.printMessage((" ------ L2 Regularization:  ", layer[8]))
                log.printMessage((" ------ Trainable: ", layer[9]))
                channelNumber = channelNumber +1
    
    
    log.startNewStep("Hidden Layers Topology")
    hiddenLayerNumber = 0
    for hiddenLayer in networkTopology[4]:
        log.printMessage(("Hidden layer Type: ", hiddenLayer[6]))      
        log.printMessage(("HiddenLayer: ", hiddenLayerNumber))        
        log.printMessage((" --- Hidden units: ", hiddenLayer[0]))
        log.printMessage((" --- Activation Function: ", hiddenLayer[1]))        
        log.printMessage((" --- Dropout: ", hiddenLayer[2]))
        log.printMessage((" --- L1 Regularization: ", hiddenLayer[3]))
        log.printMessage((" --- L2 Regularization: ", hiddenLayer[4]))
        log.printMessage((" --- Trainable: ", hiddenLayer[5]))
        hiddenLayerNumber = hiddenLayerNumber+1

    log.startNewStep("Output Layer")
    log.printMessage(("Output units: ", networkTopology[5][0]))     
    log.printMessage(("L1 Regularization: ", networkTopology[5][1]))  
    log.printMessage(("L2 Regularization: ", networkTopology[5][2]))  
    
    log.startNewStep("Training Parameters")   
    log.printMessage(("  Train the network: ", trainingParameters[0]))            
    log.printMessage(("  Use Momentum: ", trainingParameters[2]))                          
    log.printMessage(("  Maximum training epoches: ", trainingParameters[3]))
    log.printMessage(("  Batchsize: ", trainingParameters[4]))
    log.printMessage(("  Initial Learning rate: ", trainingParameters[5]))
    log.printMessage(("  Momentum: ", trainingParameters[6]))
        
    log.startNewStep("Experiment Parameters")   
    log.printMessage(("  Experiment name: ", experimentParameters[2]))            
    log.printMessage(("  Base directory: ", experimentParameters[0]))            
    log.printMessage(("  Image directory: ", experimentParameters[1]))  
    log.printMessage(("  Repetitions: ", experimentParameters[3]))                      
    log.printMessage(("  Generate Metrics: ", experimentParameters[4]))    
    log.printMessage(("  Generate Synchronized Metrics: ", experimentParameters[5]))  
    
    log.startNewStep("Saving Network Parameters")   
    
    log.printMessage(("  Saving network: ", saveNetworkParameters[0]))            
      
              
    repetitions = experimentParameters[3]   
        
    if len(networkTopology[6][0][0][3]) == 2:    
        repetitions = channel[0][3][1]
        
    nFoldPositionsSet = []
    channelsRepositories = []
    classesLabel = []
    
    for i in range(repetitions): 
                        
            time = datetime.datetime.now()                  
            
            if isTraining or isGeneratingMetrics:
                dataSet, classesLabel, nFoldPositionsSet, audioSet, channelsRepositories = DataUtil.loadData(log, baseDirectory, channels, loadImagesStrategy, nFoldPositionsSet, i, channelsRepositories, classesLabel)  
                                 
            else:
                dataSet = []  
            
#            au = 0
#            print "SHape:", numpy.array(dataSet).shape           
#            print "SHape0:", numpy.array(dataSet[0]).shape           
#            print "SHape00:", numpy.array(dataSet[0][0]).shape           
#            for img in dataSet[0][0][0]:
#                
#                cv2.imwrite("/data/datasets/SAVEE/visualization/imagesAudio/"+str(au)+".png", img)
#                au = au+1
#                
#            raw_input("here")
            
            classifier, networkState, bestValidationNetworkState, bestTestNetworkState = MCCNN.MCCNN(networkTopology, trainingParameters,experimentParameters,visualizationParameters,dataSet, i, preLoadedFilters, log)              
            
            
            if isSavingNetwork:
                log.startNewStep("Saving the Network")
                      
                saveNetworkName = modelDirectory +"/repetition_"+str(i)+"_BestTest_"+experimentName+"_.save"
                log.printMessage(("  ---Saving Best Test Network directory: ", saveNetworkName))      
                DataUtil.saveNetwork(networkTopology,trainingParameters,experimentParameters,visualizationParameters,bestTestNetworkState, saveNetworkName)
                
                saveNetworkName = modelDirectory +"/repetition_"+str(i)+"_BestValidation_"+experimentName+"_.save"
                log.printMessage(("  ---Saving Best Validation Network directory: ", saveNetworkName))      
                DataUtil.saveNetwork(networkTopology,trainingParameters,experimentParameters,visualizationParameters,bestValidationNetworkState, saveNetworkName)
                      
                      
                saveNetworkName = modelDirectory +"/repetition_"+str(i)+"_FINAL_"+experimentName+"_.save"
                log.printMessage(("  ---Saving Final Network directory: ", saveNetworkName))      
                DataUtil.saveNetwork(networkTopology,trainingParameters,experimentParameters,visualizationParameters,networkState, saveNetworkName)
            
            
            if isGeneratingMetrics:
                log.startNewStep("Metrics")
                                             
                metricsLabel = []
                for labels in range(len(classesLabel)):
                    metricsLabel.append(classesLabel[labels][1])
                
                time = (datetime.datetime.now()-time).total_seconds()
                                        
                trainingTime.append(time)                        
                trueData = []     
                testSetX, testSetY = dataSet[2]
                            
                for value in testSetY[0]:
                    trueData.append(value)                    
                
                predictedData = []
                      
                    
                for value in range(len(testSetX[0])):
                    inputs = []
                    for channel in range(len(channels)): 
                        inputStructure = channels[channel][0]                    
                        image = DataUtil.loadImage(dataSet[2],channel,inputStructure[2],value, 1, inputStructure[0], loadImagesStrategy, inputStructure[1], False)[0][0]                        
                                                
                        inputs.append(image)     
                        
                    timeR = datetime.datetime.now()
                    predictedData.append(MCCNN.classify(classifier[len(classifier)-1],inputs,batchSize)[0])    
                    timeR = (datetime.datetime.now() - timeR).total_seconds()
                    recognitionTime.append(timeR)
                  
                             
                MCCNN.getClassificationReport(trueData,predictedData,metricsFile,metricsDirectory,experimentName,i,log, metricsLabel)
                
                accuracy.append(MCCNN.getAccuracy(trueData,predictedData, metricsLabel))            
                
                precisionMicro.append(MCCNN.getPrecision(trueData,predictedData,"micro", metricsLabel))
                
                recallMicro.append(MCCNN.getRecall(trueData,predictedData,"micro", metricsLabel))
                fScoreMicro.append(MCCNN.getFScore(trueData,predictedData,"micro", metricsLabel))
                
                precisionMacro.append(MCCNN.getPrecision(trueData,predictedData,"macro", metricsLabel))
                recallMacro.append(MCCNN.getRecall(trueData,predictedData,"macro", metricsLabel))
                fScoreMacro.append(MCCNN.getFScore(trueData,predictedData,"macro", metricsLabel))
                
                precisionWeighted.append(MCCNN.getPrecision(trueData,predictedData,"weighted", metricsLabel))
                recallWeighted.append(MCCNN.getRecall(trueData,predictedData,"weighted", metricsLabel))
                fScoreWeighted.append(MCCNN.getFScore(trueData,predictedData,"weighted", metricsLabel))                  
                
                log.printMessage(("Accuracy:", accuracy[len(accuracy)-1])) 
                log.printMessage(("Training time:", time))
                                        
                classesPerSet = []
                for t in trueData: 
                    if not t in classesPerSet:
                        classesPerSet.append(t)
                        
                classesInEachSet.append(classesPerSet)
                
                precisionsPerClass.append(MCCNN.getPrecision(trueData,predictedData,None, metricsLabel))
                recallsPerClass.append(MCCNN.getRecall(trueData,predictedData,None, metricsLabel))
                fScoresPerClass.append(MCCNN.getFScore(trueData,predictedData,None, metricsLabel))

                #FOR Movement SEQUENCES FABO
                if experimentParameters[5]:
                    log.startNewStep("Synchronized Metrics")                    
                    
                    CKFolder = "/data/datasets/FABO/FABO/" 
                    
                    #labels AFEW = {"Angry":0, "Contempt":1, "Disgust":2, "Fear":3, "Happy":4, "Neutral":5, "Sad":6, "Surprise":7}
                    labels = {"ANGER":0, "ANXIETY":1, "BOREDOM":2, "DISGUST":3, "FEAR":4, "HAPPINESS":5, "NGT SRP":6, "PST SRP":7, "PUZZLEMENT":8,"SADNESS":9,"UNCERTAINTY":10}
                    trueDataS = []
                    predictedDataS = []
                    for CKc in os.listdir(CKFolder):
                        label = labels[CKc]                        
                        predictionsPerSequence = []
                        for CKs in os.listdir(CKFolder+"/"+CKc):
                            
                            numberOfFrames = networkTopology[6][0][0][0]
                            
                            #AFEW files = sorted(os.listdir(CKFolder+"/"+CKc+"/"+CKs), key=lambda x: int(x.split('.')[0].split('_')[1]))
                            files = sorted(os.listdir(CKFolder+"/"+CKc+"/"+CKs), key=lambda x: int(x.split('.')[0]))                        
                            
                            shadowImages = []
                            imageIndex = 0
                            for a in range(len(files)):
                                imagesInsequence = []
                                while not (len(files) % numberOfFrames == 0) :
                                    files.append(files[-1])
                                    
                                for ah in range(numberOfFrames):
                                    f = cv2.imread(CKFolder+"/"+CKc+"/"+CKs+"/"+files[imageIndex]) 
                                    imagesInsequence.append(f)
                                    imageIndex = imageIndex+1
                                
                                print "Geting a shadow of: ", CKFolder+"/"+CKc+"/"+CKs
                                shadowImage = ImageProcessingUtil.doConvShadow(imagesInsequence)
                                shadowImage,frame = DataUtil.prepareDataLive(shadowImage, DataUtil.DATA_MODALITY["Image"], networkTopology[6][0][0][4]) 
                                shadowImages.append(shadowImages)
                                
                                if len(shadowImages) == 3:
                                    print "I have three shadows"
                                    result = MCCNN.classify(classifier[len(classifier)-1],[shadowImages],trainingParameters[4])[0]  
                                    print "Result: ", result
                                    predictionsPerSequence.append(result)   
                                    
                                    shadowImages = []
                            
                            finalPrediction = numpy.bincount(predictionsPerSequence).argmax()   
                            predictedDataS.append(finalPrediction)
                            trueDataS.append(label)
                            print "Sequence: ", CKFolder+"/"+CKc+"/"+CKs
                            print "FinalPrediction: ", finalPrediction

                                                  
                    log.printMessage(("Predicted data:", predictedDataS)) 
                    log.printMessage(("True data:", trueDataS)) 
                    MCCNN.getClassificationReport(trueDataS,predictedDataS,metricsFile,metricsDirectory,experimentName,i,log, metricsLabel)  
                    
#                #FOR FACE SEQUENCES CK
#                if experimentParameters[5]:
#                    log.startNewStep("Synchronized Metrics")                    
#                    
#                    CKFolder = "/data/datasets/Cohn-Kanade/CKFaces/" 
#                    
#                    #labels AFEW = {"Angry":0, "Contempt":1, "Disgust":2, "Fear":3, "Happy":4, "Neutral":5, "Sad":6, "Surprise":7}
#                    labels = {"Anger":0, "Contempt":1, "Disgust":2, "Fear":3, "Happy":4, "Neutral":5, "Sad":6, "Surprise":7}
#                    trueDataS = []
#                    predictedDataS = []
#                    for CKc in os.listdir(CKFolder):
#                        label = labels[CKc]                        
#                        predictionsPerSequence = []
#                        for CKs in os.listdir(CKFolder+"/"+CKc):
#                            
#                            numberOfFrames = networkTopology[6][0][0][0]
#                            
#                            #AFEW files = sorted(os.listdir(CKFolder+"/"+CKc+"/"+CKs), key=lambda x: int(x.split('.')[0].split('_')[1]))
#                            files = sorted(os.listdir(CKFolder+"/"+CKc+"/"+CKs), key=lambda x: int(x.split('.')[0]))
#                            sizeOfVideo = len(files)
#                            while not (len(files) % numberOfFrames == 0) :
#                                files.append(files[-1])
#                             
#                            imagePosition = 0
#                            for window in range(sizeOfVideo/numberOfFrames):
#                                inputs = [] 
#                                
#                                for ah in range(numberOfFrames):
#                                    imageDirectory = CKFolder+"/"+CKc+"/"+CKs+"/"+files[imagePosition]
#                                    imagePosition = imagePosition+1
#                                    
#                                    f = cv2.imread(imageDirectory) 
#                                    img,frame = DataUtil.prepareDataLive(f, DataUtil.DATA_MODALITY["Image"], networkTopology[6][0][0][4]) 
#                                    inputs.append(img)
#                                
#                                result = MCCNN.classify(classifier[len(classifier)-1],[inputs],trainingParameters[4])[0]  
#                                predictionsPerSequence.append(result)   
#                                
##                            for CKi in os.listdir(CKFolder+"/"+CKc+"/"+CKs):
##                                inputs = []
##                                                                 
##                                f = cv2.imread(CKFolder+"/"+CKc+"/"+CKs+"/"+CKi) 
##                                img,frame = DataUtil.prepareDataLive(f, DataUtil.DATA_MODALITY["Image"], networkTopology[6][0][0][4]) 
##                                
##                                result = MCCNN.classify(classifier[len(classifier)-1],[[img,img,img,img]],trainingParameters[4])[0]  
##                                predictionsPerSequence.append(result)   
#                                
#                            finalPrediction = numpy.bincount(predictionsPerSequence).argmax()   
#                            predictedDataS.append(finalPrediction)
#                            trueDataS.append(label)
#
#                                                  
#                    log.printMessage(("Predicted data:", predictedDataS)) 
#                    log.printMessage(("True data:", trueDataS)) 
#                    MCCNN.getClassificationReport(trueDataS,predictedDataS,metricsFile,metricsDirectory,experimentName,i,log, metricsLabel)                                                                            

                    
# FOR AUDIO
#                    if experimentParameters[5]:
#                    log.startNewStep("Synchronized Metrics")                    
#
#                    predictedDataS = []
#                    trueDataS = []
#
#                    for a in range(len(audioSet[0])):
#                        label = audioSet[1][a]
#                        audio = audioSet[0][a]
#                        
#
#                                                
#                        audiosPerChannel = DataUtil.createTestInputAudio(audio, channels)
#                        
#                        if len(audiosPerChannel[0]) > 0:            
#                            trueDataS.append(label)
#                            
#                            predictionsPerAudioS = []
#                            
#       
#                            for audioIndex in range(len(audiosPerChannel[0])):   
#                                inputs = []
#                                for channel in range(len(channels)):
#                                    inputs.append(audiosPerChannel[channel][audioIndex])
#                                
#                                result = MCCNN.classify(classifier[len(classifier)-1],inputs,trainingParameters[4])[0]                                   
#                                predictionsPerAudioS.append(result)         
#                                                      
#                            finalPrediction = numpy.bincount(predictionsPerAudioS).argmax()   
#                            predictedDataS.append(finalPrediction)
#                                           
#       
#                    log.printMessage(("Predicted data:", predictedDataS)) 
#                    log.printMessage(("True data:", trueDataS)) 
#                    MCCNN.getClassificationReport(trueDataS,predictedDataS,metricsFile,metricsDirectory,experimentName,i,log, metricsLabel)
#                    
#                                      
#                    precisionSynchronized.append(MCCNN.getPrecision(trueDataS,predictedDataS,"micro", metricsLabel))
#                    recallSynchronized.append(MCCNN.getRecall(trueDataS,predictedDataS,"micro", metricsLabel))
#                    fScoreSynchronized.append(MCCNN.getFScore(trueDataS,predictedDataS,"micro", metricsLabel))
#                    accuracySynchronized.append(MCCNN.getAccuracy(trueDataS,predictedDataS, metricsLabel))
#                    
#                    precisionSynchronizedPerClass.append(MCCNN.getPrecision(trueDataS,predictedDataS,None, metricsLabel))
#                    recallSynchronizedPerClass.append(MCCNN.getRecall(trueDataS,predictedDataS,None, metricsLabel))
#                    fScoreSynchronizedPerClass.append(MCCNN.getFScore(trueDataS,predictedDataS,None, metricsLabel))
#                    
#                    log.printMessage(("Accuracy:", accuracySynchronized[len(accuracySynchronized)-1])) 
                    
#                    precisionMicro.append(MCCNN.getPrecision(trueData,predictedData,"micro", metricsLabel))
#                    
#                    recallMicro.append(MCCNN.getRecall(trueData,predictedData,"micro", metricsLabel))
#                    fScoreMicro.append(MCCNN.getFScore(trueData,predictedData,"micro", metricsLabel))
#                    
#                    precisionMacro.append(MCCNN.getPrecision(trueData,predictedData,"macro", metricsLabel))
#                    recallMacro.append(MCCNN.getRecall(trueData,predictedData,"macro", metricsLabel))
#                    fScoreMacro.append(MCCNN.getFScore(trueData,predictedData,"macro", metricsLabel))
#                    
#                    precisionWeighted.append(MCCNN.getPrecision(trueData,predictedData,"weighted", metricsLabel))
#                    recallWeighted.append(MCCNN.getRecall(trueData,predictedData,"weighted", metricsLabel))
#                    fScoreWeighted.append(MCCNN.getFScore(trueData,predictedData,"weighted", metricsLabel))                  
#                    
#                    log.printMessage(("Accuracy:", accuracy[len(accuracy)-1])) 
#                    log.printMessage(("Training time:", time))
#                                            
#                    classesPerSet = []
#                    for t in trueData:
#                        if not t in classesPerSet:
#                            classesPerSet.append(t)
#                            
#                    classesInEachSet.append(classesPerSet)
#                    
#                    precisionsPerClass.append(MCCNN.getPrecision(trueData,predictedData,None, metricsLabel))
#                    recallsPerClass.append(MCCNN.getRecall(trueData,predictedData,None, metricsLabel))
#                    fScoresPerClass.append(MCCNN.getFScore(trueData,predictedData,None, metricsLabel))
                    
                
                             
    if isGeneratingMetrics:
        averagePrecisionWeighted = numpy.mean(precisionWeighted)
        averageRecallWeighted = numpy.mean(recallWeighted)
        averageFScoreWeighted = numpy.mean(fScoreWeighted)
        
        trainingTimeAvg = numpy.mean(trainingTime)
        recognitionTimeAvg = numpy.mean(recognitionTime)
        accuracyAverage = numpy.mean(accuracy)
    
        
        
        setPrecisionPerClasses = []
        setRecallPerClasses = []
        setFScorePerClasses = []
        for c in range(outputUnits):
            setPrecisionPerClasses.append([])
            setRecallPerClasses.append([])
            setFScorePerClasses.append([])
            
        h = 0        
        for c in classesInEachSet:        
            u = 0
            for i in c:            
                setPrecisionPerClasses[i].append(precisionsPerClass[h][u])
                setRecallPerClasses[i].append(recallsPerClass[h][u])
                setFScorePerClasses[i].append(fScoresPerClass[h][u])
                u = u+1
            h = h+1
        
        
        
        averagePrecisionPerClasses = []
        averageRecallPerClasses = []
        averageFScorePerClasses = []
        
        for a in range(len(setPrecisionPerClasses)):                
            averagePrecisionPerClasses.append(numpy.mean(setPrecisionPerClasses[a]))
            averageRecallPerClasses.append(numpy.mean(setRecallPerClasses[a]))
            averageFScorePerClasses.append(numpy.mean(setFScorePerClasses[a]))
            
    
        
        log.startNewStep("Final Metrics for "+str(repetitions) + " repetitions.")  
            
        log.printMessage("Training time:" + str(trainingTimeAvg) + " -  Standard Deviation: " + str(numpy.std(trainingTime)))
        log.printMessage("Recognition time:" + str(recognitionTimeAvg) + " -  Standard Deviation: " + str(numpy.std(recognitionTime)))
    
        
        log.startNewStep("Accuracy")        
        log.printMessage("Accuracy:" + str(accuracyAverage) + " -  Standard Deviation: " + str(numpy.std(accuracy)))
        
        log.startNewStep("Precision")
        log.printMessage("Precision Weighted:" + str(averagePrecisionWeighted) + " -  Standard Deviation: " + str(numpy.std(precisionWeighted)))    
        
        log.printMessage("Precision Per class:")    
        i = 1
        for c in averagePrecisionPerClasses:
           log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
           i = i+1    
             
        log.startNewStep("Recall")
        log.printMessage("Recall Weighted:" + str(averageRecallWeighted) + " -  Standard Deviation: " + str(numpy.std(recallWeighted)))    
        
     
        log.printMessage("Recall Per class:")    
        i = 1
        for c in averageRecallPerClasses:
           log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
           i = i+1    
                      
        
        log.startNewStep("F1 Score")
        log.printMessage("F1 Score Weighted:" + str(averageFScoreWeighted) + " -  Standard Deviation: " + str(numpy.std(fScoreWeighted)))    
        
     
        log.printMessage("F1 Score Per class:")    
        i = 1
        for c in averageFScorePerClasses:
           log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
           i = i+1          
     
#        if experimentParameters[5]:
#            log.startNewStep("Synchronized Metrics After all repetitions")       
#            averagePrecisionSynchronized = numpy.mean(precisionSynchronized)
#            averageRecallSynchronized = numpy.mean(recallSynchronized)
#            averageFScoreSynchronized= numpy.mean(fScoreSynchronized)
#            
#            accuracyAverageSynchronized = numpy.mean(accuracySynchronized)
#        
#                        
#            setPrecisionPerClasses = []
#            setRecallPerClasses = []
#            setFScorePerClasses = []
#            for c in range(outputUnits):
#                setPrecisionPerClasses.append([])
#                setRecallPerClasses.append([])
#                setFScorePerClasses.append([])
#                
#            h = 0        
#            for c in classesInEachSet:        
#                u = 0
#                for i in c:            
#                    setPrecisionPerClasses[i].append(precisionSynchronizedPerClass[h][u])
#                    setRecallPerClasses[i].append(recallSynchronizedPerClass[h][u])
#                    setFScorePerClasses[i].append(fScoreSynchronizedPerClass[h][u])
#                    u = u+1
#                h = h+1
#            
#            
#            
#            averagePrecisionPerClasses = []
#            averageRecallPerClasses = []
#            averageFScorePerClasses = []
#            
#            for a in range(len(setPrecisionPerClasses)):                
#                averagePrecisionPerClasses.append(numpy.mean(setPrecisionPerClasses[a]))
#                averageRecallPerClasses.append(numpy.mean(setRecallPerClasses[a]))
#                averageFScorePerClasses.append(numpy.mean(setFScorePerClasses[a]))
#                
#        
#            
#            log.startNewStep("Final Metrics for "+str(repetitions) + " repetitions.")  
#                
#        
#            log.startNewStep("Accuracy")        
#            log.printMessage("Accuracy:" + str(accuracyAverageSynchronized) + " -  Standard Deviation: " + str(numpy.std(accuracySynchronized)))
#            
#            log.startNewStep("Precision")
#            log.printMessage("Precision Weighted:" + str(averagePrecisionSynchronized) + " -  Standard Deviation: " + str(numpy.std(precisionSynchronized)))    
#            
#            log.printMessage("Precision Per class:")    
#            i = 1
#            for c in averagePrecisionPerClasses:
#               log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
#               i = i+1    
#                 
#            log.startNewStep("Recall")
#            log.printMessage("Recall Weighted:" + str(averageRecallSynchronized) + " -  Standard Deviation: " + str(numpy.std(recallSynchronized)))    
#            
#         
#            log.printMessage("Recall Per class:")    
#            i = 1
#            for c in averageRecallPerClasses:
#               log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
#               i = i+1    
#                          
#            
#            log.startNewStep("F1 Score")
#            log.printMessage("F1 Score Weighted:" + str(averageFScoreSynchronized) + " -  Standard Deviation: " + str(numpy.std(fScoreSynchronized)))    
#            
#         
#            log.printMessage("F1 Score Per class:")    
#            i = 1
#            for c in averageFScorePerClasses:
#               log.printMessage("      Class "+str(i)+"-"+ str(metricsLabel[i-1])+": "+str(c))
#               i = i+1                           
                    
    return classifier
    
#    if(createConvFeatures):
#        log.startNewStep("Creating ConvFeatures")
#        features = MCCNN.getConvolutionalFeatures(convolutionOutputLayer,imagesDirectory,batch_size,imageSize)
#        DataUtil.writeSingleFile(features, convFeaturesDirectory+"ConvFeatures_"+experimentName+".txt")           
#        log.printMessage(("ConvFeatures created at : ", convFeaturesDirectory+"ConvFeatures_"+experimentName+".txt"))    
#    
#    if(hintonDiagrams):
#        log.startNewStep("Creating Hinton Diagrams")        
#            
#        MCCNN.createHintonDiagram(savedState,hintonDiagram,len(conParams),len(channelsTopology),len(channelsTopology)*len(conParams) +2)
#    
#    if(createOutputImages):
#            log.startNewStep("Showing OutputImages")
#            #MCNN.showOutputImages(outputConvLayers,batch_size,imagesDirectory,imageSize,channels,len(conParams))
#            log.startNewStep("Creating OutputImages")
#            outputIndex = 0
#            
#                        
#            for c in range(len(channelsTopology)):
#                                
#                if channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelX"]:
#                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input(SobelX)/"
#                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
#                    MCCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
#                    outputIndex = outputIndex+1 
#                                
#                elif channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelY"]:
#                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input(SobelY)/"
#                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
#                    MCCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
#                    outputIndex = outputIndex+1
#                else:
#                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input/"
#                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
#                    MCCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
#                    outputIndex = outputIndex+1  
#                        
#                for l in range(len(conParams)):                                    
#                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/layer"+str(l)+"/"+"Conv/"
#                    log.printMessage(("Creating Output in: ", directoryToSaveImage))
#                    MCCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
#                    outputIndex = outputIndex+1
#                    
#                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/layer"+str(l)+"/"+"MaxPooling/"
#                    log.printMessage(("Creating Output in: ", directoryToSaveImage))
#                    MCCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
#                    outputIndex = outputIndex+1
#     