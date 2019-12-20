# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

import time

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d
from theano.tensor import TensorType

import numpy
import Image

from Utils import DataUtil
from Utils import ImageProcessingUtil
import Layers

import cv2 

from sklearn.metrics import classification_report, confusion_matrix, precision_score,recall_score,f1_score,accuracy_score
import pylab

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import os
 
import datetime

def MCCNN(networkTopology, trainingParameters,experimentParameters,visualizationParameters,dataSets,currentRepetition, preLoadedFilters, log):    
                
    """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type conLayersParams: numpy.random.RandomState
        :param conLayersParams: Parameters for the convolutionalLayers. Each parameter is related to one
                                layer. Each parameter is composed by:  [numberOfFilters,receptiveFieldsX,
                                                                        receptiveFieldsY,poolSieX,poolSizeY]
        """                     
    
    log.startNewStep("Generating the Model")
    rng = numpy.random.RandomState(23455)              
    
    hiddenLayers =  networkTopology[4]
    outputLayer = networkTopology[5]
    channels = networkTopology[6]    
    loadImagesStrategy = networkTopology[7]   
    
    baseDirectory = experimentParameters[0]    
    experimentName =  experimentParameters[2]
    experimentDirectory = baseDirectory + "/experiments/"+experimentName+"/"
    metricsDirectory = experimentDirectory+"/metrics/trainingEpoches/"
    modelDirectory=experimentDirectory+"/model/"
    
    filtersTrainingDirectory =  experimentDirectory+"/Filters/trainingHistory/Repetition_"+str(currentRepetition)+"/"    
    filtersLiveDirectory =  experimentDirectory+"/Filters/liveFeatures/Repetition_"+str(currentRepetition)+"/"    
    
    isTraining = trainingParameters[0]
    usingDropOut = trainingParameters[1]
    isUsingMomentum = trainingParameters[2]
    numberEpoches = trainingParameters[3]
    batchSize = trainingParameters[4]    
    inititalLearningRate = trainingParameters[5]   
    initial_momentum = trainingParameters[6]   
    L1Reg = trainingParameters[9]   
    L2Reg = trainingParameters[10]
    isUsingL1 = trainingParameters[11]
    isUsingL2 = trainingParameters[12]

    isVisualizingFilters = visualizationParameters[3]
    isVisualizingEpoches = visualizationParameters [4]
        
    inputs = []
    finalInputs = []    
    
    dropOutIndex = T.iscalar('dropOutIndex')    
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
                                                          
    channelLayers = []    
    channelOutputShapes = []
    
    isUsingLSTM = False    
    for hiddenLayer in hiddenLayers:
        if hiddenLayer[6] == DataUtil.HIDDEN_LAYER_TYPE["LSTM"]:
            isUsingLSTM = True
            break
        
    if isUsingLSTM:    
        #x.reshape((batchSize, numberInputFirstLayer,inputImageSize[0],inputImageSize[1]))
        tensor5Dimensions = TensorType('float64', (False,)*5)
        LSTMInput = tensor5Dimensions("LSTMInput")
            
    for channel in range(len(channels)):
             
            inputStructure = channels[channel][0]
            layers = channels[channel][1]
            
            log.printMessage(("Channel: ", channel))         

            x = T.matrix("x_"+str(channel))
            
            inputImageSize =inputStructure[4]
            inputImages = inputStructure[0]            
                        
            if isUsingLSTM:                
                inputStructure[2] = DataUtil.IMAGE_STRUCTURE["Static"]
                
            imageStructure = inputStructure[2]                    
                        
                
            numberInputFirstLayer = 3                                    
            
            if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"] or isUsingLSTM :
                x = x.reshape((batchSize, numberInputFirstLayer,inputImageSize[0],inputImageSize[1]))
            elif imageStructure == DataUtil.IMAGE_STRUCTURE["Sequence"]:
                x = x.reshape((batchSize, inputImages, numberInputFirstLayer,inputImageSize[0],inputImageSize[1]))            
                
            inputLayer = Layers.InputLayer(x, inputStructure,  batchSize)                 
            
            inputs.append(x)   
            finalInputs.append(x)                                    
            
            inputImageSize = inputStructure[4][0],inputStructure[4][1]
            inputImagesPerChannel = inputStructure[0]
                                                
            layerInput = inputLayer.output                                        
            
            convLayers = []        
            isPreviousLayerASequence = False
            inputImageShape = inputLayer.outputShape
                                
            for layer in range(len(layers)):
                log.printMessage(("---Layer: ", layer))
                log.printMessage(("-----Input shape: ", inputImageShape[len(inputImageShape)-2], ",", inputImageShape[len(inputImageShape)-1]))                 
                                
                layerType = layers[layer][5]                 
                featureMaps = layers[layer][0]                
                featureMapsDimension = layers[layer][1]                              
                poolSize = layers[layer][3]                              
                
                usingInhibition = layers[layer][4]
                usingPooling = layers[layer][2]     
                
                activationFunction = layers[layer][6]     
#                                                
#                if isPreviousLayerASequence:      
#                      print "Reshaping the output"
#                      layerInput = layerInput.reshape((batchSize, inputImageShape[1], inputImageSize[0],inputImageSize[1]))
#                      isPreviousLayerASequence = False
                                                                        
                if len(inputImageShape) == 5:
                    filtersShape = (featureMaps, inputImagesPerChannel ,inputImageShape[2],featureMapsDimension[0], featureMapsDimension[1] )                                          
                    isPreviousLayerASequence = True
                else:
                    filtersShape = (featureMaps, inputImageShape[1],featureMapsDimension[0], featureMapsDimension[1] )                                                    
                
                #print "InputShape:", inputImageShape
                
                if preLoadedFilters:
                    filters = preLoadedFilters[0][channel][layer][0]
                    if not usingInhibition == None:
                        filtersInhibition = preLoadedFilters[0][channel][layer][1]
                    else:
                        filtersInhibition = None
                else:
                    filters = None
                    filtersInhibition = None
                

                convLayer = Layers.ConvLayer(activationFunction,layerType,  usingInhibition, usingPooling, rng,filters,filtersInhibition ,input=layerInput,image_shape=inputImageShape, filter_shape=filtersShape, poolsize=(poolSize[0], poolSize[1]))
                convLayer.getOutput()      
                
                
                #validationMiniBatch = DataUtil.loadImage(dataSets[0],channel,inputStructure[2],0, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1], isUsingLSTM)[0]
#                
                #validationMiniBatch =  numpy.array(validationMiniBatch)
                #print "Shape of the images:", numpy.array(validationMiniBatch).shape
                #print "Input Shape:", inputImageShape
#                #print "Filter Shape:", filtersShape
                #inputModel = theano.function(inputs, convLayer.output)
##                     
                #test = inputModel(validationMiniBatch)     
##               # 
                #print "Shape Output:", numpy.array(test).shape
#                
#                raw_input("A")
                inputImageShape = convLayer.outputShape
                                
                log.printMessage(("-----Output shape: ", inputImageShape[len(inputImageShape)-2], ",", inputImageShape[len(inputImageShape)-1]))                 
                                
                layerInput = convLayer.output   

                layers[layer][0] = inputImageShape[1]                                                                                                                    
                                            
                convLayers.append(convLayer)
                
            channelLayers.append(convLayers)
            convLayers = []
            channelOutputShapes.append((inputImageShape[2],inputImageShape[3]))
     
    crossChannels = networkTopology[9]
    crossChannelLayers = []
    
    if not crossChannels == []:        
        log.printMessage(("CrossConvolutional layers"))  
        
        crossChannelOutputShapes = []
        for crossChannel in range(len(crossChannels)):
            log.printMessage(("Channel:",crossChannel ))                         
            
            crossChannelInputs = crossChannels[crossChannel][0]
            layers = crossChannels[crossChannel][1]
            
            layerInput = []
            
            inputImages = 0
            inputImageSize = (0,0)
            for channelInCrossChannel in range(len(crossChannelInputs)):
                channel = crossChannelInputs[channelInCrossChannel][0]
                layer = crossChannelInputs[channelInCrossChannel][1]
            
                log.printMessage(("--- Input Channel:",channel," Layer:", layer ))                
            
                inputImages += channelLayers[channel][layer].outputShape[1]
                inputImageSize = (channelLayers[channel][layer].outputShape[2], channelLayers[channel][layer].outputShape[3])
                
                layerInput.append(channelLayers[channel][layer].output)
                
            layerInput = theano.tensor.concatenate(layerInput,1)
            inputImageShape = (batchSize, inputImages, inputImageSize[0],inputImageSize[1])
            
            crossChannelConvLayers = []
            for layer in range(len(layers)):
                log.printMessage(("---Layer: ", layer))
                log.printMessage(("-----Input shape: ", inputImageShape[len(inputImageShape)-1], ",", inputImageShape[len(inputImageShape)-2]))                 
                
                layerType = layers[layer][5]                 
                featureMaps = layers[layer][0]                
                featureMapsDimension = layers[layer][1]                              
                poolSize = layers[layer][3]                              
                
                usingInhibition = layers[layer][4]
                usingPooling = layers[layer][2]     
                
                activationFunction = layers[layer][6]     
                                                
                if isPreviousLayerASequence:                      
                      layerInput = layerInput.reshape((batchSize, inputImageShape[1], inputImageSize[0],inputImageSize[1]))
                      isPreviousLayerASequence = False
                                                                        
                if len(inputImageShape) == 5:
                    filtersShape = (featureMaps, inputImagesPerChannel ,inputImageShape[1],featureMapsDimension[0], featureMapsDimension[1] )                      
                    isPreviousLayerASequence = True
                else:
                    filtersShape = (featureMaps, inputImageShape[1],featureMapsDimension[0], featureMapsDimension[1] )                                                    
            
                if preLoadedFilters:
                    #print "SHape:", numpy.array(preLoadedFilters[1][crossChannel][layer])
                    
                    filters = preLoadedFilters[1][crossChannel][layer]
                    if not usingInhibition == None:
                        filtersInhibition = preLoadedFilters[1][crossChannel][layer][1]
                    else:
                        filtersInhibition = None
                else:
                    filters = None
                    filtersInhibition = None
                
    
                convLayer = Layers.ConvLayer(activationFunction,layerType,  usingInhibition, usingPooling, rng,filters,filtersInhibition ,input=layerInput,image_shape=inputImageShape, filter_shape=filtersShape, poolsize=(poolSize[0], poolSize[1]))                    
                convLayer.getOutput()                
 
                inputImageShape = convLayer.outputShape
                                
                log.printMessage(("-----Output shape: ", inputImageShape[len(inputImageShape)-1], ",", inputImageShape[len(inputImageShape)-2]))                 
                                
                layerInput = convLayer.output   
    
                layers[layer][0] = inputImageShape[1]                                                                                                                      
                                            
                crossChannelConvLayers.append(convLayer)
            crossChannelLayers.append(crossChannelConvLayers)
            convLayers = []
            crossChannelOutputShapes.append((inputImageShape[2],inputImageShape[3]))
               
    outputs = []
    hiddenLayerInputShape = 0
    log.printMessage(("Hidden layer"))  
    
    if not crossChannels == []:
        for c in range(len(crossChannelLayers)):            
            layer = crossChannelLayers[c]
            output = layer[len(layer)-1].output.flatten(2)
            outputs.append(output)      
            
            lastLayer = crossChannels[c][1]            
            filtersLastLayer = lastLayer[len(lastLayer)-1][0]
            hiddenLayerInputShape = hiddenLayerInputShape + filtersLastLayer*crossChannelOutputShapes[c][0]*crossChannelOutputShapes[c][1]     
                
    for c in range(len(channels)):
            if crossChannels == [] or ( not crossChannels == [] and channels[c][2]):
                layer = channelLayers[c]
                output = layer[len(layer)-1].output.flatten(2)
                outputs.append(output) 

                
                lastLayer = channels[c][1]

                filtersLastLayer = lastLayer[len(lastLayer)-1][0]
#                print "Channel:", c
#                print "---lasteLayer:", lastLayer
#                print "---Filters Last Layer:", filtersLastLayer
#                print "---Output of this layer:", filtersLastLayer*channelOutputShapes[c][0]*channelOutputShapes[c][1]                 
                hiddenLayerInputShape = hiddenLayerInputShape + filtersLastLayer*channelOutputShapes[c][0]*channelOutputShapes[c][1]                 
#                print "hidden Layer Channel", c, ":", hiddenLayerInputShape
    
  
                        
    convLayersOutput = theano.tensor.concatenate(outputs,1)       
    
    modelHiddenLayers = []
    hiddenLayerInput = convLayersOutput
    
    
    hiddenLayerNumber = 0    
    for hiddenLayer in hiddenLayers:
        log.printMessage(("Hidden Layer: ",hiddenLayerNumber))         
        log.printMessage(("Hidden Layer Type: ",hiddenLayer[6]))      
        log.printMessage(("---Number Inputs: ",hiddenLayerInputShape)) 
        log.printMessage(("---Number Outputs: ",hiddenLayer[0])) 
        log.printMessage(("---Use Dropout: ",hiddenLayer[2])) 
        
        outputUnits = hiddenLayer[0]
        activationFunction = hiddenLayer[1]
        if not preLoadedFilters ==None:        
                filters = preLoadedFilters[2][hiddenLayerNumber]
                
        if hiddenLayer[6] == DataUtil.HIDDEN_LAYER_TYPE["Common"]:
            useDropOut = hiddenLayer[2]                    
                        
            
                
            if useDropOut:                              
                    hiddenLayer = Layers.DropoutHiddenLayer(activationFunction, rng,dropOutIndex,  filters, input=hiddenLayerInput, n_in=hiddenLayerInputShape,
                                 n_out=hiddenLayer[0])    
            else:
                    hiddenLayer = Layers.HiddenLayer(activationFunction, rng, filters, input=hiddenLayerInput, n_in=hiddenLayerInputShape,
                                     n_out=hiddenLayer[0]   )      
        
        elif hiddenLayer[6] == DataUtil.HIDDEN_LAYER_TYPE["LSTM"]:
            #validationMiniBatch = DataUtil.loadImage(dataSets[0],channel,inputStructure[2],0, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1])[0]
            #validationMiniBatch =  numpy.array(validationMiniBatch)
            #print "Shape:", numpy.array(validationMiniBatch).shape
            #print "Input Shape:", inputImageShape
            #print "Filter Shape:", filtersShape
            #raw_input("A")
        
                
            hiddenLayer = Layers.LSTMLayer(LSTMInput, hiddenLayerInputShape, hiddenLayer[0], outputLayer[0], hiddenLayerInput, activationFunction, channelLayers, channels, filters)        
            

            #test_model = theano.function([LSTMInput], hiddenLayer.output)
            #raw_input("B")
                 
            #test = test_model(validationMiniBatch)     
            #raw_input("C")
            
            #print "Shape Output:", numpy.array(test).shape
            #print "Output:", test
            
            #raw_input("D")
                                  
        hiddenLayerInputShape = outputUnits
        hiddenLayerInput = hiddenLayer.output
        modelHiddenLayers.append(hiddenLayer)
        hiddenLayerNumber = hiddenLayerNumber+1 
    
    log.printMessage(("Output Layer"))                      
    
    #raw_input("here0")
    if not preLoadedFilters ==None:        
            filters = preLoadedFilters[3]
            #print "filterss:", filters					            
												
												
    #print "Filters:", filters        
    classifierLayer = Layers.LogisticRegression(filters, input=modelHiddenLayers[len(modelHiddenLayers)-1].output, n_in=hiddenLayers[len(hiddenLayers)-1][0], n_out=outputLayer[0])
    
    log.printMessage(( "Inputs: ", hiddenLayers[len(hiddenLayers)-1][0] ))
    log.printMessage(( "Outputs: ", outputLayer[0] ))
      
    bestValidationNetwork = None 
    bestTestNetwork = None 
    
    if isTraining:
        log.startNewStep("Training")
        trainSet,validationSet, testSet = dataSets
                    
        # compute number of minibatches for training, validation and testing        
        n_train_batches = len(trainSet[0][0])
        n_valid_batches = len(validationSet[0][0])
        n_test_batches = len(testSet[0][0])
        n_train_batches /= batchSize
        n_valid_batches /= batchSize
        n_test_batches /= batchSize        
        
        L1Norm = 0
        L2Sqr = 0
        
        isUsingL1 = False
        isUsingL2 = False
        
        
        if not crossChannels == []:  
            for channel in range(len(crossChannels)):
                for layer in range(len(crossChannels[channel][1])):                
                    useL1 = channels[channel][1][layer][7]
                    useL2 = channels[channel][1][layer][8]                    
                    usingInhibition = crossChannels[channel][1][layer][4]                
                    
                    if useL1:
                        L1Norm = L1Norm + abs((crossChannelLayers[channel][layer].W).sum())
                        if usingInhibition != None:
                            L1Norm = L1Norm + abs((crossChannelLayers[channel][layer].wInhibitory).sum())
                        
                        isUsingL1 = True
                        
                    if useL2:    
                        L2Sqr = L2Sqr + (crossChannelLayers[channel][layer].W**2).sum()                    
                        if usingInhibition != None:
                            L2Sqr = L2Sqr + (crossChannelLayers[channel][layer].wInhibitory**2).sum()    
                        isUsingL2 = True
            
        for channel in range(len(channels)):
            if crossChannels == [] or ( not crossChannels == [] and channels[channel][2]):
                for layer in range(len(channels[channel][1])):                
                    useL1 = channels[channel][1][layer][7]
                    useL2 = channels[channel][1][layer][8]
                    usingInhibition = channels[channel][1][layer][4]                
                    
                    if useL1:
                        L1Norm = L1Norm + abs((channelLayers[channel][layer].W).sum())
                        if usingInhibition != None:
                            L1Norm = L1Norm + abs((channelLayers[channel][layer].wInhibitory).sum())
                        
                        isUsingL1 = True
                        
                    if useL2:    
                        L2Sqr = L2Sqr + (channelLayers[channel][layer].W**2).sum()                    
                        if usingInhibition != None:
                            L2Sqr = L2Sqr + (channelLayers[channel][layer].wInhibitory**2).sum()    
                        isUsingL2 = True            
            
        for hiddenLayer in range(len(hiddenLayers)):
           useL1 = hiddenLayers[hiddenLayer][2]
           useL2 = hiddenLayers[hiddenLayer][3]
           
           if hiddenLayers[hiddenLayer][6] == DataUtil.HIDDEN_LAYER_TYPE["LSTM"]:
               if useL1:
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_xi).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_hi).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_ci).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_xf).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_hf).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_cf).sum())                        
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_xc).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_hc).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_xo).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_ho).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_co).sum())                        
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].c0).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W_hy).sum())
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].b_y).sum())
                        isUsingL1 = True
               if useL2:                            
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_xi).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_hi).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_ci).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_xf).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_hf).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_cf).sum()                        
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_xc).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_hc).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_xo).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_ho).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_co).sum()                        
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].c0).sum()
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W_hy).sum()                        
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].b_y).sum()
                        isUsingL2 = True
               cost =  (modelHiddenLayers[hiddenLayer].cost)     
           else:
               if useL1:
                        L1Norm = L1Norm + abs((modelHiddenLayers[hiddenLayer].W).sum())
                        isUsingL1 = True
               if useL2:    
                        L2Sqr = L2Sqr + (modelHiddenLayers[hiddenLayer].W**2).sum()   
                        isUsingL2 = True
        
        
               if outputLayer[1] :
                    L1Norm = L1Norm + abs((classifierLayer.W).sum())
                    isUsingL1 = True
                    
               if outputLayer[2] :    
                    L2Sqr = L2Sqr + (classifierLayer.W**2).sum()   
                    isUsingL2 = True
            
               cost =  (classifierLayer.negative_log_likelihood(y))     
        
        if isUsingL1:
            cost = cost + numpy.cast[theano.config.floatX](L1Reg) * L1Norm
        if isUsingL2:
            cost = cost + numpy.cast[theano.config.floatX](L2Reg) * L2Sqr
        
        interactiveLearningRate = theano.shared(numpy.cast[theano.config.floatX](inititalLearningRate) )
        
        assert initial_momentum >= 0. and initial_momentum < 1. 
        
        log.printMessage(( "Using momentum:", isUsingMomentum, ": ", initial_momentum ))
        momentum =theano.shared(numpy.cast[theano.config.floatX](initial_momentum), name='momentum')
                
        params = []                  
        
        if not crossChannels == []:  
#            print "Compiling cross channel params"
            for channel in range(len(crossChannelLayers)):            
                for layer in range(len(crossChannelLayers[channel])):   
    
                        isTrainable = crossChannels[channel][1][layer][9]
    
                        usingInhibition = crossChannels[channel][1][layer][4]
                        
                        if usingInhibition != None and usingInhibition[1]:
                            params += crossChannelLayers[channel][layer].paramsInhibition            
                        if isTrainable:
                            params += crossChannelLayers[channel][layer].params            
                             
        for channel in range(len(channelLayers)):            
            if crossChannels == [] or ( not crossChannels == [] and channels[channel][2]):
                for layer in range(len(channelLayers[channel])):   
    
                        isTrainable = channels[channel][1][layer][9]
    
                        usingInhibition = channels[channel][1][layer][4]
                        
                        if usingInhibition != None and usingInhibition[1]:
                            params += channelLayers[channel][layer].paramsInhibition            
                        if isTrainable:
                            params += channelLayers[channel][layer].params            
                    
        for hiddenLayer in range(len(hiddenLayers)):
             isTrainable = hiddenLayers[hiddenLayer][5]
             if isTrainable:
                 params += modelHiddenLayers[hiddenLayer].params
        if not isUsingLSTM:        
            params += classifierLayer.params
        
#        grads = T.grad(cost, params)
        
        log.printMessage(( "Compiling cost function: "))
        grads = []
        parameterPosition = 0  
        for param in params:			
			parameterPosition = parameterPosition+1   
			log.printMessage(( "Parameter: ", parameterPosition, "(", len(params),")"))
			gparam = T.grad(cost, param)
			grads.append(gparam)

        updates = []  
        for param_i, grad_i in zip(params, grads):
            if isUsingMomentum:
                param_update = theano.shared(param_i.get_value()*numpy.cast[theano.config.floatX](0.))              
                updates.append((param_i, param_i - interactiveLearningRate * param_update)) 
                updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*grad_i)) 
            else:
            
                updates.append((param_i, param_i - interactiveLearningRate * grad_i))            
   
        usingDropOut = False        
        for hiddenLayer in range(len(hiddenLayers)):            
            if hiddenLayers[hiddenLayer][2]:
                usingDropOut = True
                    
        if usingDropOut:            
            givensTrain = {dropOutIndex: numpy.cast['int32'](1)}
            givensTestAndValidation = {dropOutIndex: numpy.cast['int32'](0)}
        else:            
            givensTrain = {}
            givensTestAndValidation = {}
        
        log.printMessage(( "Compiling network functions", "...Wait!"))
        testInputs = []
        
        if not isUsingLSTM:            
            testInputs  = inputs 
            testInputs.append(y)
            test_model = theano.function(testInputs, classifierLayer.errors(y),
                     givens=givensTestAndValidation)
                                     
            validate_model = theano.function(testInputs, classifierLayer.errors(y),
                    givens=givensTestAndValidation) 
                    
            train_model = theano.function(testInputs, cost, updates=updates,
                  givens=givensTrain)                        
        else:            
            
            for channel in range(len(channelLayers)):
                testInputs.append(LSTMInput)
                
            testInputs.append(modelHiddenLayers[-1].target)
            train_model = theano.function(testInputs, cost, updates=updates)      
                             
            validate_model = theano.function(testInputs, modelHiddenLayers[-1].errors(modelHiddenLayers[-1].target)) 
                    
            test_model = theano.function(testInputs, modelHiddenLayers[-1].errors(modelHiddenLayers[-1].target))        

        #print "Result:", result
        #raw_input("here")    
        log.startNewStep("Training Model")      
        # early-stopping parameters
        patience = 999999999  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
       
        #if isUsingLSTM:
        #    validation_frequency = 1
        #else:
        validation_frequency = min(n_train_batches, patience / 2)
        
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        last_validations = []
        best_validation_loss = numpy.inf
        best_test_loss = numpy.inf
        test_score = 0.
    
        epoch = 0
        done_looping = False  
        
        updatesLearningRate = 0
        testScores0 = 0        
                
        if isVisualizingEpoches:
            validationScores = []
            testingScores = []   
            trainingScores = []
            plt.ion()    
            plt.clf()
            imageName = "repetition_"+str(currentRepetition)+".png"
                    
        if isVisualizingFilters[2] or isVisualizingFilters[3]:                        
            liveFilters = createEmptyLiveImage(channels)
            if not crossChannels == []:
                liveFiltersCross = createEmptyLiveImage(crossChannels)
                     
        if isVisualizingFilters[0] or isVisualizingFilters[1]:
            filterImages  = createEmptyLiveImage(channels)
            if not crossChannels == []:
                filterImagesCross  = createEmptyLiveImage(crossChannels)
        
        bestValidationLoss = 100    
        besTestResult = 0
        
        while (epoch < numberEpoches) and (not done_looping):
            epoch = epoch + 1
                        
            if epoch == 1 or epoch % 10 == 0:
                if isVisualizingFilters[0] or isVisualizingFilters[1]:                                
                                    
                    emptyOneEpochVisualization = createEmptyLiveImage(channels)
                   
                    
                    currentFilterImage= getLiveFiltersImage(channelLayers, emptyOneEpochVisualization, channels, "conv")    
                   
                    if epoch % 10 == 0:
                       visualizationEpoch = (epoch / 10) +1
                    else:
                        visualizationEpoch = epoch
                    filterImages = createFilterImages(currentFilterImage, filterImages, channels, visualizationEpoch,)                    
                   
                    
                    displayFilters(channels, filterImages, isVisualizingFilters[0], (isVisualizingFilters[1], filtersTrainingDirectory), False, "conv")   
                   
                    
                    if not crossChannels == []:
                         emptyOneEpochVisualizationCross = createEmptyLiveImage(crossChannels)
                         currentFilterImageCross= getLiveFiltersImage(crossChannelLayers, emptyOneEpochVisualizationCross, crossChannels, "cross")    
                         filterImagesCross = createFilterImages(currentFilterImageCross, filterImagesCross, crossChannels, visualizationEpoch)    
                         displayFilters(crossChannels, filterImagesCross, isVisualizingFilters[0], (isVisualizingFilters[1], filtersTrainingDirectory), False, "cross")   
                                        
                            
            for minibatch_index in xrange(n_train_batches):
                
                iter = (epoch - 1) * n_train_batches + minibatch_index
            
#                train_losses = []
#                for i in xrange(n_train_batches):

#                   trainMiniBatchX = []

#                   for channel in range(len(channels)):
#                      inputStructure = channels[channel][0]

#                      trainMiniBatch = DataUtil.loadImage(trainSet,channel,inputStructure[2],i, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1], isUsingLSTM)
#                      trainMiniBatchX.append(trainMiniBatch[0])
#                      trainMiniBatchY = trainMiniBatch[1]

#                   trainMiniBatchX.append(trainMiniBatchY)

#                   train_losses.append(validate_model(*trainMiniBatchX))


#                this_train_loss = numpy.mean(train_losses)

#                log.printMessage((" Epoch:",epoch, " minibatch ",minibatch_index + 1,"/",n_train_batches, " . Train Error: ", this_train_loss * 100. ))

    
                trainMiniBatchX = []                
                
                for channel in range(len(channels)):
                    inputStructure = channels[channel][0]                    
                    trainMiniBatch = DataUtil.loadImage(trainSet,channel,inputStructure[2],minibatch_index, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1], isUsingLSTM)
                    trainMiniBatchX.append(trainMiniBatch[0])     
                    trainMiniBatchY = trainMiniBatch[1]
                    

                trainMiniBatchX.append(trainMiniBatchY)                

                train_model(*trainMiniBatchX)
                
                if usingDropOut:
                    paramsHidden = []
                    for hiddenLayer in range(len(hiddenLayers)):
                         isTrainable = hiddenLayers[hiddenLayer][5]
                         if isTrainable:
                             paramsHidden.append(modelHiddenLayers[hiddenLayer].params[0])
#                    raw_input("here")         
                    Layers.rescale_weights(paramsHidden, 15.)
         
                if isVisualizingFilters[2] or isVisualizingFilters[3]:
                     liveFilters = getLiveFiltersImage(channelLayers, liveFilters, channels, "conv")                     
                         
                     displayFilters(channels, liveFilters, isVisualizingFilters[2], (isVisualizingFilters[3], filtersLiveDirectory), True, "conv")      

                     
                     if not crossChannels == []:
                         liveFiltersCross = getLiveFiltersImage(crossChannelLayers, liveFiltersCross, crossChannels, "cross")
                         displayFilters(crossChannels, liveFiltersCross, isVisualizingFilters[2], (isVisualizingFilters[3], filtersLiveDirectory), True, "cross")      

                if (iter + 1) % validation_frequency == 0:                                            
   
                    #validation_losses = []
                    #for i in xrange(n_valid_batches):
                        
                    train_losses = []
                    for i in xrange(n_train_batches):

                        trainMiniBatchX = []

                        for channel in range(len(channels)):
                          inputStructure = channels[channel][0]

                          trainMiniBatch = DataUtil.loadImage(trainSet,channel,inputStructure[2],i, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1], isUsingLSTM)
                          trainMiniBatchX.append(trainMiniBatch[0])
                          trainMiniBatchY = trainMiniBatch[1]

                        trainMiniBatchX.append(trainMiniBatchY)

                        train_losses.append(validate_model(*trainMiniBatchX))


                    this_train_loss = numpy.mean(train_losses)

                    log.printMessage((" Epoch:",epoch, ". Train Error: ", this_train_loss * 100. ))


 #                  trainMiniBatchX = []
 
                    validation_losses = []
                    for i in xrange(n_valid_batches):


                       validationMiniBatchX = []                        
                        
                       for channel in range(len(channels)):                                             
                            inputStructure = channels[channel][0]
                            
                            validationMiniBatch = DataUtil.loadImage(validationSet,channel,inputStructure[2],i, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1], isUsingLSTM)
                            validationMiniBatchX.append(validationMiniBatch[0])     
                            validationMiniBatchY = validationMiniBatch[1]
                                                
                       validationMiniBatchX.append(validationMiniBatchY)
                        
                       validation_losses.append(validate_model(*validationMiniBatchX))                        
                                                                
                                         
                    this_validation_loss = numpy.mean(validation_losses)                     

                    log.printMessage((" Epoch:",epoch, ". Validation Error: ", this_validation_loss * 100. ))                    
                         
                    if len(last_validations) == 15:
                        last_validations.pop(0)  

                    last_validations.append(this_validation_loss)
                                        
                    unchangedValidations = 0
                    last = last_validations[0]                    
                    for ah in last_validations:                        
                        if last == ah:
                            unchangedValidations = unchangedValidations+1
                        else:   
                             unchangedValidations = unchangedValidations-1
                        last = ah
                        
                    if isUsingMomentum:    
                        if momentum.get_value() < 0.99:
                            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
                            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
#                    # adaption of learning rate                            

                    
                    #print "Last validations:", last_validations
                    #print "UnchangedValidations: ", unchangedValidations                   
                    if unchangedValidations >= 5:
                        updatesLearningRate = updatesLearningRate + 1
                        log.printMessage(("Update learning rate --- from: ", interactiveLearningRate.get_value(), " to: ", (interactiveLearningRate.get_value() * 0.985)))
                        #print "Update learning rate --- from: ", interactiveLearningRate.get_value(), " to: ", (interactiveLearningRate.get_value() * 0.985)
                        inititalLearningRate = inititalLearningRate /2
                        new_learning_rate = interactiveLearningRate.get_value() * 0.985
                        interactiveLearningRate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
                        last_validations = []
                        
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                           #if isUsingLSTM:
                           #    patience = patience     
                           #else:
                           patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        
                        bestValidationNetwork = DataUtil.createNetworkState(channelLayers, modelHiddenLayers, crossChannelLayers, classifierLayer, channels, crossChannels)         
                        
                        # test it on the test set
    
                    test_losses = []
                    for i in xrange(n_test_batches):
                        
                        testMiniBatchX = []                        
                        
                        for channel in range(len(channels)):
                            inputStructure = channels[channel][0]
                            testMiniBatch = DataUtil.loadImage(testSet,channel,inputStructure[2],i, batchSize, inputStructure[0], loadImagesStrategy, inputStructure[1],isUsingLSTM)
                            testMiniBatchX.append(testMiniBatch[0])     
                            testMiniBatchY = testMiniBatch[1]
                                                                            
                        testMiniBatchX.append(testMiniBatchY)
                        
                                                
                        test_losses.append(test_model(*testMiniBatchX))
                        
                    
                    #test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    
                    if test_score < best_test_loss:
                        best_test_loss = test_score
                        bestTestNetwork = DataUtil.createNetworkState(channelLayers, modelHiddenLayers, crossChannelLayers, classifierLayer, channels, crossChannels)         
                        
                    log.printMessage((" Epoch:",epoch, " . Test Error: ", test_score * 100. ))
                    log.printMessage(("----------------------------------------------------------"))
                    
                    if test_score == 0:
                        testScores0 = testScores0 + 1
                    else:
                        testScores0 = 0
                  
                    if isVisualizingEpoches:
                        validationScores.append(this_validation_loss*100)                        
                        testingScores.append(test_score*100)
                        trainingScores.append(this_train_loss*100)             
                        lineValidation, = plt.plot(range(epoch),validationScores, label="Validation Set")
                        lineTesting, = plt.plot(range(epoch), testingScores, label="Testing Set")
                        lineTraining = plt.plot(range(epoch), trainingScores, label ="Training Set")
                        
                        
                        plt.setp(lineValidation, color='r', linewidth=2.0)
                        plt.setp(lineTesting, color='b', linewidth=2.0)
                        plt.setp(lineTraining, color='y', linewidth=2.0)
                                                
                        #plt.legend([lineValidation, lineTesting, lineTraining], ["Validation Set", "Testing Set", "Training Set"])
                        plt.legend([lineValidation, lineTesting], ["Validation Set", "Testing Set"])
                        plt.xlabel("Epoches")
                        plt.ylabel("Training Scores")
                        plt.title("Visualizing Training")
                        plt.axis([0, epoch, 0, 100])
                        plt.yticks(numpy.arange(0,100, 5))
   
                        #blue_patch = mpatches.Patch(color='blue', label='Test Set')
                        #plt.legend(handles=[red_patch])
                        
                        plt.draw()       
                        plt.savefig(metricsDirectory+"/"+imageName)
                        plt.pause(0.001)
                                        
                
                if testScores0 == 5:
                   done_looping = True
                   break
                   
                if updatesLearningRate == 10:
                    done_looping = True
                    break
                
                if patience <= iter:
                    done_looping = True
                    break
        
        
    model = []        
        
    #model.append(theano.function(inputs=inputs,outputs=inputLayer.output) )
    if not isUsingLSTM:
                 
        for channel in range(len(channelLayers)):
            for layer in channelLayers[channel]:                   
                model.append(theano.function([finalInputs[channel]],layer.output))  

                  
        if not crossChannels == []:          
            for channel in range(len(crossChannelLayers)):
                crossChannelInputs = []            
                channelsInCrossChannels = crossChannels[channel][0]
                for channelInCrossChannel in range(len(channelsInCrossChannels)):
                    channelInput = channelsInCrossChannels[channelInCrossChannel][0]                
                    crossChannelInputs.append(finalInputs[channelInput])            
                    
                for layer in crossChannelLayers[channel]:                   
                    model.append(theano.function(crossChannelInputs,layer.output))                

    givensTrain = {}        
    givensTestAndValidation = {}    
    

    for hiddenLayer in range(len(hiddenLayers)):            

        model.append(theano.function(finalInputs,modelHiddenLayers[hiddenLayer].input, givens=givensTestAndValidation))         								
								
        if hiddenLayers[hiddenLayer][2]:
            givensTrain = {dropOutIndex: numpy.cast['int32'](1)}
            givensTestAndValidation = {dropOutIndex: numpy.cast['int32'](0)}   
            
        if hiddenLayers[hiddenLayer][6] == DataUtil.HIDDEN_LAYER_TYPE["LSTM"]:
            finalInputs = []
            for channel in range(len(channelLayers)):
                finalInputs.append(LSTMInput)
            
        
        model.append(theano.function(finalInputs,modelHiddenLayers[hiddenLayer].output, givens=givensTestAndValidation))
    
    model.append(theano.function(finalInputs,classifierLayer.p_y_given_x, givens=givensTestAndValidation))                        
    model.append(theano.function(finalInputs,classifierLayer.y_pred, givens=givensTestAndValidation))
    
												
    #print  "filters: ", classifierLayer.W.eval()        
    #raw_input("here")
                
    return (model, DataUtil.createNetworkState(channelLayers, modelHiddenLayers, crossChannelLayers, classifierLayer, channels, crossChannels), bestValidationNetwork, bestTestNetwork)        
    
 
def classify(predictModel, inputs,batchSize):

    inputsModel = []

    for channel in range(len(inputs)):    
        newArray = []
        
        for i in range(batchSize):
            newArray.append(inputs[channel])    
                    
        inputsModel.append(newArray)  
        print "Input size:",  numpy.array(inputsModel).shape
    return predictModel(*inputsModel)
                               
def getClassificationReport(trueData,predictData,directory,metricsDirectory,experimentName,repetition,log, metricsLabel):
         
     #print "Calculating Metrics...."     
     metrics =  (classification_report(trueData,predictData, target_names=metricsLabel))    
     cM = confusion_matrix(trueData,predictData)  
     log.startNewStep("Confusion Matrix")
     log.printMessage(cM)
     pylab.matshow(cM)
     pylab.title('Confusion matrix')
     pylab.colorbar()
     pylab.ylabel('True label')
     pylab.xlabel('Predicted label')
     metricsDirectory = metricsDirectory + "/ConfusionMatrix/"
     DataUtil.createFolder(metricsDirectory)
     pylab.savefig(metricsDirectory+"/repetition_"+str(repetition)+".png")    
         
     log.printMessage(metrics)
  
def getPrecision(trueData, predictData,average, metricsLabel):
    return precision_score(trueData,predictData,average=average)
    
def getAccuracy(trueData, predictData, metricsLabel):
    return accuracy_score(trueData,predictData,normalize= True)    
    
def getRecall(trueData, predictData,average, metricsLabel):
    return recall_score(trueData,predictData,average=average)
    
def getFScore(trueData, predictData,average, metricsLabel):
    return f1_score(trueData,predictData,average=average)       


def getConvolutionalFeatures(outputModel,directoryImages, batchSize,imageSize):
    
    convFeatures = []
    classesPath = os.listdir(directoryImages)          
    classNumber = 0
    for classs in classesPath:  
        files = os.listdir(directoryImages+os.sep+classs+os.sep)                    
        for image in files:
            imageFeatures = []
            img = cv2.imread(directoryImages+os.sep+classs+os.sep+image)
                                
            features = ImageProcessingUtil.grayImage(img,imageSize ,False,"")
            features = features.view(numpy.ndarray)
            features.shape = -1
            cFeature = classify(outputModel,features,batchSize)[0]
            #print "Shape Cfeature:", cFeature.shape
            for f in cFeature:                                    
                imageFeatures.append(f)
            imageFeatures.append(classNumber)            
            
            convFeatures.append(imageFeatures)            
        classNumber = classNumber+1
    return convFeatures


def getOutputImage(outputModels,batchSize, img,imageSize,channels,layers):
    
    grayScale = ImageProcessingUtil.grayImage(img,imageSize ,False,"")
    #grayScale = ImageProcessingUtil.resize(img,imageSize )
    
    features = grayScale.view(numpy.ndarray)
    features.shape = -1    
    
    outputIndex = 0      
    pylab.gray()
    fig = plt.figure()       
    fig.subplots_adjust(left=0.3, wspace=0.3, hspace=0.3)

    ax1 = fig.add_subplot(1+layers*2,channels,1)                
    ax1.imshow(grayScale[:, :])                
    ax1.set_title("C1: Gray")
    ax1.axis('off')                

    ax1 = fig.add_subplot(1+layers*2,channels,2)                
    sobelX = classify(outputModels[outputIndex],features,batchSize)[0][0]
    ax1.imshow(sobelX[:, :])                
    ax1.set_title("C2: SX")
    ax1.axis('off')         

    ax1 = fig.add_subplot(1+layers*2,channels,3)
    outputIndex = outputIndex+1
    sobelY = classify(outputModels[outputIndex],features,batchSize)[0][0]
    ax1.imshow(sobelY[:, :])                
    ax1.set_title("C3: SY")
    ax1.axis('off')                         
    outputIndex = outputIndex+1
    for c in range(channels):
        layersConvOut = []
        layersMaxOut = []                    
        for l in range(layers):       
           convImages = classify(outputModels[outputIndex],features,batchSize)[0]
           outputIndex = outputIndex+1
           maxPoolingImages = classify(outputModels[outputIndex],features,batchSize)[0]
           outputIndex = outputIndex+1
           
           layerConvOut = None
           layerMaxOut = None
           for i in range(len(convImages)):
               img = convImages[i]
               if layerConvOut == None:
                   layerConvOut = img
               else:    
                   layerConvOut = numpy.hstack((layerConvOut,img))
               
               img = maxPoolingImages[i]
               if layerMaxOut == None:
                   layerMaxOut = img
               else:    
                   layerMaxOut = numpy.hstack((layerMaxOut,img))
                   
           layersConvOut.append(layerConvOut)
           layersMaxOut.append(layerMaxOut)
        posConv = channels+c+1                    
        posMax = 2*channels+c+1
        for l in range(layers):
            #print str(1+layers*2)+","+str(channels)+","+str(posConv)
            ax1 = fig.add_subplot(1+layers*2,channels,posConv)
            ax1.imshow(layersConvOut[l][:, :])
            ax1.set_title("C:"+ str(c)+ "_L:"+str(l)+" Conv")
            ax1.axis('off')                    
            
            #print str(1+layers*2)+","+str(channels)+","+str(posMax)
            ax1 = fig.add_subplot(1+layers*2,channels,posMax)
            ax1.imshow(layersMaxOut[l][:, :])                        
            ax1.set_title("C:"+ str(c)+ "_L:"+str(l)+" Max")
            ax1.axis('off')                        
            
            posConv = posConv+channels*2
            posMax = posMax+channels*2    
    return ImageProcessingUtil.fig2data(fig) 
                

def showOutputImages(outputModels,batchSize,directoryImages,imageSize,channels,layers):
    directory = directoryImages    
    classesPath = os.listdir(directory)            
        
    for classs in classesPath:      
            files = os.listdir(directory+os.sep+classs+os.sep)                        
            for image in files:
                img = cv2.imread(directory+os.sep+classs+os.sep+image)
                                
                fig = getOutputImage(outputModels,batchSize, img,imageSize,channels,layers)                
                cv2.imshow('image',fig)
                #cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/Cambridge/100x100MotionWithShadows_SET1/Final.png", fig)
                key = cv2.waitKey(20)
                #break
            #break        
    cv2.destroyAllWindows()
    
def createOutputImagesSequence(outputModel, batchSize, directoryImages, directorySave, imageSize):
    
    directory = directoryImages
    featuresDirectory = directorySave       
    classesPath = os.listdir(directory)            
    gesture = 0
    
    for classs in classesPath:      
        sequencesNumber = 0                  
        sequences = os.listdir(directory+os.sep+classs+os.sep)
        for sequence in sequences:
            files = os.listdir(directory+os.sep+classs+os.sep+sequence+os.sep)            
            #files = os.listdir(directory+os.sep+classs+os.sep)            
            frameNumber = 0
            
            images = []
            for image in files:                                    
                img = cv2.imread(directory+os.sep+classs+os.sep+sequence+os.sep+image)
                #img = cv2.imread(directory+os.sep+classs+os.sep+image)                   
                              
                features = ImageProcessingUtil.resize(img,imageSize)  
                features = features.view(numpy.ndarray)
                                
                features = ImageProcessingUtil.grayImage(features,imageSize ,False,"")
                features = ImageProcessingUtil.whiten(features)         
                
                features.shape = -1
              
                frameNumber = frameNumber+1  
                images.append(features)
                                            
                
            cFeature = classify(outputModel,images,batchSize)
            
            
            cFeature = cFeature[0]
            #print "Shape cFeature:", numpy.array(cFeature).shape
            if len(numpy.array(cFeature).shape) == 4:
                if not numpy.array(cFeature).shape[0] == 1:
                    cFeature = numpy.swapaxes(cFeature, 0,1)
                cFeature = cFeature[0]   
                    
                
            
            o = 0
            DataUtil.createFolder(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep)
            #print "Shape cFeature:", numpy.array(cFeature).shape
            for im in cFeature:        
                
                img = im
                
               # print img                                      
                img = Image.fromarray(ImageProcessingUtil.convertFloatImage(img),"L") #Image.fromarray(img, "L")
                #img = Image.fromarray(img,"L")
                pylab.gray()
                                    
                #if not os.path.exists(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep): os.makedirs(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep)            
                
                
                #img.save(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg") 
                #print "Image Name: ", featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+image+"_"+str(o)+"_.jpeg"
                pylab.imsave(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+image+"_"+str(o)+"_.jpeg", im)
                #img.save(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg")      
                o= o+1                    
                
            sequencesNumber = sequencesNumber +1
            
            if sequencesNumber >3:
                break          
     
               
            gesture = gesture+1    
    
def createOutputImages(outputModel,batchSize, directoryImages, directorySave, imageSize, inputType):
        
    if  inputType == DataUtil.INPUT_TYPE["3D"]:
        createOutputImagesSequence(outputModel,batchSize, directoryImages, directorySave, imageSize)
    else:
        directory = directoryImages
        featuresDirectory = directorySave       
        classesPath = os.listdir(directory)            
        gesture = 0
        
        for classs in classesPath:                        
            #sequences = os.listdir(directory+os.sep+classs+os.sep)
            #for sequence in sequences:
               #files = os.listdir(directory+os.sep+classs+os.sep+sequence+os.sep)            
                files = os.listdir(directory+os.sep+classs+os.sep)            
                frameNumber = 0
                imagesNumber = 0
                for image in files:                                    
                    #img = cv2.imread(directory+os.sep+classs+os.sep+sequence+os.sep+image)
                    img = cv2.imread(directory+os.sep+classs+os.sep+image)                   
                    #cv2.imwrite("/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/test/test1.jpg",img)              
                    
                    features = ImageProcessingUtil.resize(img,imageSize)
                    #cv2.imwrite("/informatik2/wtm/home/barros/demo_ws/src/dialog/scripts/networkPosture/test/test2.jpg",features)              
                    
                    features = features.view(numpy.ndarray)
                    
                    if inputType == DataUtil.INPUT_TYPE["Color"]:
                        features = numpy.swapaxes(features, 2,1)
                        features = numpy.swapaxes(features, 0,1)
                        
                        features = numpy.reshape(features, (3, imageSize[0]*imageSize[1]))
                        
                    else:                
                        features = ImageProcessingUtil.grayImage(features,imageSize ,False,"")
                        
                        features = ImageProcessingUtil.whiten(features)         
                                      
                        
                        features.shape = -1
                    
                    frameNumber = frameNumber+1                                          
                    
                    cFeature = classify(outputModel,features,batchSize)
                                    
                    cFeature = cFeature[0]
                    
                    o = 0
                    DataUtil.createFolder(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep)
                    for im in cFeature:                    
                        img = im
                        
                        img = numpy.array(img)                        
                        
                        f = open(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.txt","w")                        
                        
                       # print "Image:", img
                        for ll in img:
                            f.write(str(ll)+"\n")
                        f.close()
                                                
                       # print img                                      
                        
                        img = ImageProcessingUtil.convertFloatImage(img)
                        

                        
                        
                        img = Image.fromarray(img,"L") #Image.fromarray(img, "L")
                        #img = Image.fromarray(img,"L")
                         
                        pylab.gray()
                                            
                        #if not os.path.exists(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep): os.makedirs(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep)            
                        
                        
                        #img.save(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg") 
                        
                        pylab.imsave(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg", im)
                        #img.save(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg")      
                        o= o+1                    
                        
                    imagesNumber = imagesNumber +1
                    
                    if imagesNumber >3:
                        break                  
                   
                gesture = gesture+1    
             


def createHintonDiagram(directory, saveDirectory,convLayers,channels,totalLayers):    
        
    lines = 5
    rows = 0
       
    loadedParams = DataUtil.loadState(directory,totalLayers)
        
    index = 0
    
    for channel in range(channels):
        
        for layer in range(convLayers):
            layerImage = []
            for a in range(len(loadedParams[index][0].get_value())):                
                                
                f = loadedParams[index][0].get_value()[a][0].astype(theano.config.floatX)
                img = ImageProcessingUtil.convertFloatImage(f)
                #img = Image.fromarray(f, "L")
                #pylab.imsave(finalDirectory, img)
                layerImage.append(img)
                finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)+"/"
                DataUtil.createFolder(finalDirectory)  
                finalDirectory = finalDirectory +"_filter_"+str(a)+"_"+str(index)+".png"
                #img = ImageProcessingUtil.grayImage(img)
                matplotlib.image.imsave(finalDirectory,img, cmap=matplotlib.cm.gray)                
                #plt.imshow(img)
               # plt.savefig(finalDirectory)
                #img.save(finalDirectory)
                
                
                #DataUtil.saveHintonDiagram(loadedParams[index][0].get_value()[f][0],finalDirectory)
            rows = (len(layerImage)/lines)
            
            kernelSize = Image.fromarray(layerImage[0]).size
            #print kernelSize
            new_im = Image.new('RGB', (lines*kernelSize[0],rows*kernelSize[1]))
            imgIndex = 0
            for i in range(lines):
                for j in range(rows):                
                    #paste the image at location i,j:
                    posL = kernelSize[0]*i 
                    posR = kernelSize[0]*j 
                    new_im.paste(Image.fromarray(layerImage[imgIndex]), (posL,posR))
                    imgIndex = imgIndex+1
                    
            finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)            
            DataUtil.createFolder(finalDirectory)    
            finalDirectory = finalDirectory +"_filter_"+str(index)+".png"      
            new_im.save(finalDirectory)   
            #plt.imshow(new_im)
            #plt.savefig(finalDirectory)
            #new_im = ImageProcessingUtil.convertFloatImage(new_im)
            #matplotlib.image.imsave(finalDirectory,new_im)                
            index = index+1                   
    
def displayFilters(channels, filtersMapsImages, show, save, live, name):
    
    #print "General Shape:", numpy.array(filtersMapsImages).shape
    for channel in range(len(channels)):            
        for layer in range(len(channels[channel][1])):              
              img = numpy.array(filtersMapsImages[channel][layer])
              if live:
                  live = "Live"
              else:
                  live=""
              if show:                    
                   cv2.imshow(name+"-"+"--C: "+ str(channel+1)+ " L: "+ str(layer+1),  img)           
              if save[0]:
                  if live:
                      saveFolder = save[1] + "/"+name+"/"+"C"+str(channel+1)+"_L"+str(layer+1)
                      
                      fileName = str(datetime.datetime.now())+".png"
                      
                      DataUtil.createFolder(saveFolder)
                      cv2.imwrite(saveFolder+"/"+fileName, img)
                  else:
                      DataUtil.createFolder(save[1]+ "/"+name+"/")
                      cv2.imwrite(save[1]+ "/"+name+"/"+"--C: "+ str(channel+1)+ " L: "+ str(layer+1)+".png", img)
              
              key = cv2.waitKey(20)                                 
                        

#def getAllFiltersImage(filtersMapsImages, )

def createFilterImages(currentFilterImage,filterImages,  channels, epoch):
         
    #filterImages = [] 
    newImages = createEmptyVisualizingImage(channels,epoch)
    
    for channel in range(len(channels)):
        #channelImages = []
        for layer in range(len(channels[channel][1])):            
            newImages[channel][layer].paste(filterImages[channel][layer],(0,0))                 
            usingInhibition = not(channels[channel][1][layer][4] == None)
            pX = 55*(epoch-1)
            if usingInhibition:
                pX = 55*(epoch-1)*3
                
            newImages[channel][layer].paste(currentFilterImage[channel][layer],(pX,0))             
    
    return newImages
            

def getLiveFiltersImage(channelLayers, filtersMapsImages, channels, name):
    
    for channel in range(len(channelLayers)):                        
        for layer in range(len(channelLayers[channel])):                                    
            usingInhibition = not(channels[channel][1][layer][4] == None)
                        
            
            filterMaps = channelLayers[channel][layer].params[0].get_value(borrow=True)  
            if usingInhibition:
                filterMapsInhibition = channelLayers[channel][layer].paramsInhibition[0].get_value(borrow=True)   
            else: 
                filterMapsInhibition = filterMaps
                          
            
            img = ImageProcessingUtil.convertFloatImage(filterMaps)
            posX = 0
            posY = -55
            if name == "cross":
                imageColorSpace = DataUtil.FEATURE_TYPE["GrayScale"]          
            else:
                imageColorSpace = channels[channel][0][6]                   
            for w, wi in zip(filterMaps, filterMapsInhibition):                    
                posY = posY + 55                                               
                if imageColorSpace == DataUtil.FEATURE_TYPE["RGB"] and layer==0:                          
                    if channels[channel][0][2] == DataUtil.IMAGE_STRUCTURE["Sequence"]:
                        w = w[0]
                        wi = wi[0]
                        
                    red = ImageProcessingUtil.convertFloatImage(w[0])
                    green = ImageProcessingUtil.convertFloatImage(w[1])
                    blue = ImageProcessingUtil.convertFloatImage(w[2])
                    img = numpy.array([red, green, blue])
                    img = numpy.swapaxes(img, 0,1) 
                    img = numpy.swapaxes(img,1,2) 
                    
                    convert = "RGB"
                    
                    if usingInhibition:
                        imgF = w / (channelLayers[channel][layer].decayTerm + wi)
                        
                        redI = ImageProcessingUtil.convertFloatImage(wi[0])
                        greenI = ImageProcessingUtil.convertFloatImage(wi[1])
                        blueI = ImageProcessingUtil.convertFloatImage(wi[2])
                        
                        redF = ImageProcessingUtil.convertFloatImage(imgF[0])
                        greenF = ImageProcessingUtil.convertFloatImage(imgF[1])
                        blueF = ImageProcessingUtil.convertFloatImage(imgF[2])
                        
                        imgI = numpy.array([redI, greenI, blueI])
                        imgI = numpy.swapaxes(img, 0,1) 
                        imgI = numpy.swapaxes(imgI,1,2)  
                                             
                        imgF = numpy.array([redF, greenF, blueF])
                        imgF = numpy.swapaxes(img, 0,1)
                        imgF = numpy.swapaxes(imgF,1,2)  
                        
                        
                    
                else:
                    img = ImageProcessingUtil.convertFloatImage(w[0])
                    if usingInhibition:
                        imgF = ImageProcessingUtil.convertFloatImage(w / (channelLayers[channel][layer].decayTerm + wi))[0]                        
                        imgI = ImageProcessingUtil.convertFloatImage(wi[0])
                        
                    convert = "L"
                    
                img = ImageProcessingUtil.resizeInterpolation(img,(50,50), 0)                                    
                img = Image.fromarray(img.astype('uint8')).convert(convert)
                
                filtersMapsImages[channel][layer].paste(img,(posX,posY))
                
                if usingInhibition:                    
                    posIX = posX + 52
                    posFX = posIX + 52
                    
                    imgI = ImageProcessingUtil.resizeInterpolation(imgI,(50,50), 0)                                    
                    imgI = Image.fromarray(imgI.astype('uint8')).convert(convert)
                    
                    imgF = ImageProcessingUtil.resizeInterpolation(imgF,(50,50), 0)                                    
                    imgF = Image.fromarray(imgF.astype('uint8')).convert(convert)
                    
                    filtersMapsImages[channel][layer].paste(imgI,(posIX,posY))
                    filtersMapsImages[channel][layer].paste(imgF,(posFX,posY))
                    


    return filtersMapsImages

def createEmptyVisualizingImage(channels, epoches):
    
    liveFitlersPerChannel = []
    for channel in channels:
        channelFilters = []
        for layer in channel[1]:  
            usingInhibition = not(layer[4] == None)
            featureMaps = layer[0]
            rows = 55*featureMaps                        
            collumns = 55*epoches
            if usingInhibition:
                collumns = 55*epoches*3
            image = Image.new('RGB', (collumns,rows))
            channelFilters.append(image)
        liveFitlersPerChannel.append(channelFilters)
                
    return liveFitlersPerChannel        
                
        
def createEmptyLiveImage(channels):
    
    liveFitlersPerChannel = []
    for channel in channels:
        channelFilters = []
        for layer in channel[1]:               
            usingInhibition = not(layer[4] == None)
            featureMaps = layer[0]
            rows = 55*featureMaps                                    
            collumns = 55
            if usingInhibition :
                collumns = 55*3
            image = Image.new('RGB', (collumns,rows))
            channelFilters.append(image)
        liveFitlersPerChannel.append(channelFilters)
                
    return liveFitlersPerChannel
    
    #loadedParams = DataUtil.loadState(directory,totalLayers)
        
#    index = 0
#    
#    for channel in range(channels):
#        
#        for layer in range(convLayers):
#            for f in range(len(loadedParams[index][0].get_value())):                
#                finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)
#                DataUtil.createFolder(finalDirectory)
#                
#                finalDirectory = finalDirectory +"/filter_"+str(f)+".png"                
#                DataUtil.saveHintonDiagram(loadedParams[index][0].get_value()[f][0],finalDirectory)
#                   
#            index = index+1
#    
#    finalDirectory = saveDirectory + "Hidden"+"/"
#    DataUtil.createFolder(finalDirectory)    
#    finalDirectory = finalDirectory + "hiddenLayer.png"    
#    DataUtil.saveHintonDiagram(loadedParams[index][0].get_value(),finalDirectory)
#    index = index+1
#    
#    finalDirectory = saveDirectory + "Output"+"/"
#    DataUtil.createFolder(finalDirectory)    
#    finalDirectory = finalDirectory + "outputLayer.png"    
#    DataUtil.saveHintonDiagram(loadedParams[index][0].get_value(),finalDirectory)


"""
trueData = []
for i in range(6):
    for h in range(30):
        trueData.append(i)
 
trueData = numpy.array(trueData)
#trueData = numpy.ravel(trueData)
#print trueData.shape
       
predictData = [0,4,5,0,5,0,0,0,0,0,5,5,0,0,0,5,0,0,5,0,5,0,4,0,0,4,0,0,4,0,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,3,1,1,1,3,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,3,3,1,3,1,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,0,3,3,4,3,4,4,4,4,4,4,4,1,3,4,4,4,4,3,4,1,4,1,4,1,4,4,4,5,5,5,5,5,5,5,4,5,0,5,4,5,5,5,5,5,5,5,5,5,4,4,5,5,4,5,5,4,5]
predictData = numpy.array(predictData)
#predictData = numpy.ravel(predictData)
#print predictData.shape
cM = confusion_matrix(trueData,predictData)  
     
labels = ["Circle", "P. Left", "P. Right", "Stand", "Stop", "Turn"]    


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cM, cmap=pylab.cm.Greys)
pylab.title('Confusion matrix')

for i, cas in enumerate(cM):
    for j, c in enumerate(cas):
        if c>0:
            plt.text(j-.2, i+.2, c, fontsize=14, color='red')

fig.colorbar(cax)
pylab.gray()
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pylab.ylabel('True label')
pylab.xlabel('Predicted label')
pylab.savefig("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/NimbroLiveTest/Turn/_confusionMatrix.png")    

pylab.matshow(cM)
pylab.title('Confusion matrix')
pylab.colorbar()
pylab.ylabel('True label')
pylab.xlabel('Predicted label')
pylab.set_xticklabels([''] + labels)
pylab.set_yticklabels([''] + labels)
pylab.savefig("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/NimbroLiveTest/Turn/_confusionMatrix.png")    

print getFScore(trueData,predictData,"micro")
print getFScore(trueData,predictData,None)
print cM
#print f1_score(trueData,trueData,"micro")  
"""
