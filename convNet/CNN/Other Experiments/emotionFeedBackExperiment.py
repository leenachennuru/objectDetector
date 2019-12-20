# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
    
from Utils import DataUtil
from Utils import ImageProcessingUtil
from Networks import MCCNN

import MCCNNExperiments
import os
import numpy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

import cv2


def Experiment():
                
    #Experiment parameters
    baseDirectory = "/export/experiments/emotionalFeedback/"        
    
    experimentName = "Cohn-Kanade-feedback-Demo"    
    repetitions = 5     
    isGeneratingMetrics = True
    
    experimentParameters = []
    experimentParameters.append(baseDirectory)
    experimentParameters.append("-")  
    
    experimentParameters.append(experimentName) 
    experimentParameters.append(repetitions)
    experimentParameters.append(isGeneratingMetrics) 
    
    
    #Network Topology    
    
    loadImagesStrategy = DataUtil.LOAD_IMAGES_STRATEGY["LoadAll"]
    
    
    
    #None
    preloadedFilters = None #DataUtil.loadNetworkState("/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/experiments/Test_New_Framework/model/repetition_0_Test_New_Framework_.save")[4]
            
    """ 
    The layer is composed of nine parameters: (numberOfFeatureMaps, dimensionFeatureMaps, isUsingeMaxPooling, dimensionMaxPooling,isUsingShunthingInhibition, layerType, activationFunction, L1Reg, L2Reg, Trainable)
    
    int numberOfFeatureMaps - Number of filter maps in that layer.
    (int,int) dimensionFeatureMaps - dimension of the feature maps in the layer.
    bool useMaxPooling - flag that defines the use MaxPooling in the layer
    (int, int) dimensionMaxPooling - dimension of the maxpooling operation.
    inhibitoryField - defines if the layers is using or not an inhibitory field.
    LAYER_TYPE - defines what kind of filters will be used in ths layer.
    ACTIVATION_FUNCTION - defines which is the activation function used in each convLayer
    (bool) L1Regularization - defines if the layer is present on the L1Regularization strategy
    (bool) L2Regularization - defines if the layer is present on the L2Regularization strategy
    (bool) TrainFilters - defines if this layer will be trained
    DATA_MODALITY - Modality that this channel will deal with
     
    """
    
    """    Inhibitory field:
            (bool,LAYER_TYPE, trainable) isUsingShunthingInhibition - flag that defines if the layer is using shunting inhibition ( http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1699868 ).
                                                                Defines also which type of inhibition is used.
                                                                Defines also if this lateral inhibition will be trained or not.
    """                                                                
    inhibitoryField0 =  (DataUtil.LAYER_TYPE["Common"], True)
            
    
    layer01 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]        
    
    layer11 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]        
        
    layer2 = [20,(7,7), True, (4,4), None, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]            
    layer2i = [20,(7,7), True, (2,2), inhibitoryField0, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]            
    
    layer3 = [20,(5,5), True, (2,2), None, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]            
    
    #layer2 = [1,(3,3), False, (2,2), inhibitoryField0, DataUtil.LAYER_TYPE["SobelX"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False]            
    
    #layer3 = [1,(3,3), False, (2,2), None, DataUtil.LAYER_TYPE["SobelY"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False]            
        

    """     
    Channels Topology
    The channel topology is composed of two parameters: (colorSpace, imageStructue)
    inputStructure  - represents which inputStructure this channel uses.        
    Layers - Layers that compose this channel    
    bool - attachThisChannel - Indicate if this channel is attached to the hidden layer (Use False when the channel is used as input to a crossChannel)                
    """ 
    
    
    """ Input Structure
        (int,DATA_MODALITY, IMAGE_STRUCTURE, String, (int,int)) inputStructure - Defines what is the structure of the input. It is possible to use different modalities of input,
                                                              with different configurations. Each input has a number of inputImages, a data type and a data structure.
                                                               
                      int - inputImages
                      DATA_MODALITY  - dataModality
                      IMAGE_STRUCTURE - imageStructure
                      String - data Directory inside the baseDirectory
                      (int,int) - imageSize 
                      bool - useDataAugmentation (Only works when load image strategy is LoadAll) 
                      COLOR_SPACE - input color space
                      imagePosition - If the image folder contains a sequence structure ("StaticInSequence"), but the channel will deal with only one image,
                                      identify here which image in the sequence the channel will deal with.
                      
    """     
     
    inputStructure0 = [4, DataUtil.DATA_MODALITY["Face"], DataUtil.IMAGE_STRUCTURE["Static"], "cohn-kanade+_Feedback_", (60,60), True, DataUtil.COLOR_SPACE["Whiten"], 0]        
    
    inputStructure1 = [4, DataUtil.DATA_MODALITY["Face"], DataUtil.IMAGE_STRUCTURE["StaticInSequence"], "cohn-kanade-Feedback-4-frames", (64,49), False, DataUtil.COLOR_SPACE["GrayScale"], 3]    
    #inputStructure0 = [2, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Sequence"], "sequences_newCode", (50,50), False, DataUtil.COLOR_SPACE["GrayScale"], 0]    
    #inputStructure1 = [2, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["sequences_basic_emotions"], "sequences_newCode", (50,50), False, DataUtil.COLOR_SPACE["GrayScale"], 1]
    
    
    channel0 = (inputStructure0, [layer01, layer2], False)            
    channel1 = (inputStructure1, [layer11, layer2i ], False)       
      
     
    channels = [] 
     
    channels.append(channel0)
    #channels.append(channel1) 
    #channels.append(channel2)
    #channels.append(channel3)
 

  
    """ Cross Channels
    The CrossChannels topology is composed of two parameters: (inputChannels, layers)
    (int,int[])  - represents which layer from which channels will serve as input to the crossChannelLayers
             
    Layers - Layers that compose this Crosschannel
    
    The input channels must have the same shape(the same image size)
    """ 
      
    
    crossConvolutionChannels = []
    
    crossConvChannel1 = [ [(0,1), (1,1)], [layer3]] 
    
    crossConvChannel2 = [ [(0,0), (1,0)], [layer3]]
    
    #crossConvolutionChannels.append(crossConvChannel1)
    #crossConvolutionChannels.append(crossConvChannel2)

    """ 
    Each hidden layer is composed of 6 parameters: (numberOfHiddenUnits, ActivationFunction, UseDroput, L1Reg, L2Reg, Trainable )
    
    int numberOfHiddenUnits - Number of hidden Units in this layer.        
    ACTIVATION_FUNCTION - defines which is the activation function used in each convLayer
    (bool) useDropout - defines if this layer will use dropout technique
    (bool) L1Regularization - defines if the layer is present on the L1Regularization strategy
    (bool) L2Regularization - defines if the layer is present on the L2Regularization strategy
    (bool) Trainable - defines if this layer will be trained
    
    """ 
        
    hiddenLayer1 = [100, DataUtil.ACTIVATION_FUNCTION["Tanh"], False, True, False, True]
    hiddenLayer2 = [50, DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False, True]
    
    
    hiddenLayers = [] 
    hiddenLayers.append(hiddenLayer1)
    #hiddenLayers.append(hiddenLayer2)
 
    """ 
    Each output layer is composed of 3 parameters: (numberOfOutputUnits, L1Reg, L2Reg)
    
    int numberOfOutputUnits - Number of output Units in this layer.            
    (bool) L1Regularization - defines if the layer is present on the L1Regularization strategy
    (bool) L2Regularization - defines if the layer is present on the L2Regularization strategy    
    
    """
        
    outputLayer = [3, True, False]

    networkTopology = []
    networkTopology.append("-")
    networkTopology.append("-")    
    networkTopology.append("-")
    networkTopology.append("-")
    networkTopology.append(hiddenLayers)
    networkTopology.append(outputLayer)    
        
    networkTopology.append(channels)
    
    networkTopology.append(loadImagesStrategy)
    networkTopology.append(preloadedFilters)
    
    networkTopology.append(crossConvolutionChannels)
        
    
    #Training parameters
    isTraining = True    
    
    """ 
    Training strategies
    """
    
    isUsingMomentum = False
        
    maxTrainingEpochs = 200
    L1Regularization = 0.001
    L2Regularization = 0.0001
    
    """
    Training Parameters 
    """
    batchSize = 20
    inititalLearningrate = 0.1
    momentum = 0.99
      
    """
    Training data parameters 
    (int, int, int) dataSetDivision = porcentage of data for training, validation and testing respectively.
    """
    dataSetDivision = (60,20,20)    
    
    
    trainingParameters = []
    trainingParameters.append(isTraining)    
    trainingParameters.append("") 
    trainingParameters.append(isUsingMomentum)
    trainingParameters.append(maxTrainingEpochs)
    trainingParameters.append(batchSize)
    trainingParameters.append(inititalLearningrate)
    trainingParameters.append(momentum)
    trainingParameters.append(dataSetDivision)
    trainingParameters.append("")
    trainingParameters.append(L1Regularization) 
    trainingParameters.append(L2Regularization)
    trainingParameters.append("")
    trainingParameters.append("")
        
    #Visualization parameters
    
    isCreatingHintongDiagrams = False #""" TO DO """
    isCreatingOutputImages    = False #""" TO DO """
    isCreatingConvFeatures = False #""" TO DO """
    isVisualizingTrainingEpoches = True
    """ 
    (bool, bool, bool, bool) isVisualizingTrain - The first parameter is related to visualize the filters change during training, the second one is to save the filters
    change as an image. The third one creates a live visualization of the filters during training. The forth one creates video from the filters updates.
    """
    visualizeFiltersTraining = False 
    saveFilters= True
    visualizeFiltersLive = False
    createVideoFiltersVisualization = False #""" TO DO """

    isVisualizingFilters = (visualizeFiltersTraining, saveFilters, visualizeFiltersLive, createVideoFiltersVisualization)
           
    visualizationParameters = []
    visualizationParameters.append(isCreatingHintongDiagrams)
    visualizationParameters.append(isCreatingOutputImages)
    visualizationParameters.append(isCreatingConvFeatures)
    visualizationParameters.append(isVisualizingFilters)
    visualizationParameters.append(isVisualizingTrainingEpoches)
    

    """ 
    (bool) isSavingNetwork - The first parameter is related to save the network after it is trained. The second parameter load a 
                                            network that was previously saved.
        
    """
    saveNetwork = True         
    
    saveNetworkParameters = [saveNetwork]

    
    return MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )
    
    
    
def testJaffeSet():
    
    #modelDirectory = "/export/experiments/emotionalFeedback/experiments/Cohn-Kanade_feedback/model/repetition_3_Cohn-Kanade_feedback_.save"
    #modelDirectory = "/export/experiments/emotionalFeedback//experiments/Cohn-Kanade-feedback-Demo//model//repetition_0_Cohn-Kanade-feedback-Demo_.save"
    modelDirectory =  "/export/experiments/emotionalFeedback//experiments/Cohn-Kanade-feedback-Demo//model//repetition_3_Cohn-Kanade-feedback-Demo_.save"
    
    networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectory)       
            
    saveNetworkParameters = [False]
    
    
    network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )
        
    imageDirectory = "/export/experiments/emotionalFeedback/JAFFE_TEST/"    
    
    plotNeutral = []
    plotNegative = []
    plotPositive = []
    plt.ion()   
    imageNumber = 0
    
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0) 
    if vc.isOpened(): # try to get the first frame
        rval, f = vc.read()
     #   rval2, f2 = vc2.read()
    else:
        rval = False
 
    while rval:     
        #frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        imageNumber = imageNumber +1     
        img,frame = DataUtil.prepareDataLive(f, DataUtil.DATA_MODALITY["Face"], networkTopology[6][0][0][4])            
        
        cv2.imshow("preview3", frame)        
        
        
        result = MCCNN.classify(network[len(network)-2],[img],trainingParameters[4])[0]*100
        predicted = numpy.argmax(numpy.array(result), axis=0)
        negative = numpy.array(result)[0]
        neutral = numpy.array(result)[2]
        positive = numpy.array(result)[1]
        
#            if predicted == 0:
#                positive = 0
#                negative = numpy.array(result)[predicted]
#                neutral = 0
#            elif predicted == 1:
#                positive = 0
#                negative = 0
#                neutral =  numpy.array(result)[predicted]
#            else:
#                positive = numpy.array(result)[predicted]
#                negative = 0
#                neutral =  0
                
        if len(plotNeutral) == 10:
            del plotNeutral[0]
            del plotNegative[0]
            del plotPositive[0]
            imageNumber = 10
            plt.cla()
            
        plotNeutral.append(neutral)
        plotNegative.append(negative)
        plotPositive.append(positive)

        
        linePositive, = plt.plot(numpy.arange(imageNumber),plotPositive, label="Positive")
        lineNegative, = plt.plot(numpy.arange(imageNumber), plotNegative, label="Negative")
        lineNeutral, = plt.plot(numpy.arange(imageNumber), plotNeutral, label="Neutral")
        
        
        plt.setp(linePositive, color='r', linewidth=2.0)
        plt.setp(lineNegative, color='b', linewidth=2.0)
        plt.setp(lineNeutral, color='y', linewidth=2.0)
                                
        plt.legend([linePositive, lineNegative, lineNeutral], ["Positive", "Negative", "Neutral"])
        plt.xlabel("Images")
        plt.ylabel("Probability")
        plt.title("Visualizing Faces")
        plt.axis([0, imageNumber, -50, 150])
        plt.yticks(numpy.arange(0,100, 5))
   
        #blue_patch = mpatches.Patch(color='blue', label='Test Set')
        #plt.legend(handles=[red_patch])
        
        plt.draw()          
        #img = cv2.imread(imageDirectory+"/"+classes+"/"+image)
        #cv2.imshow("Face", img)
        plt.pause(0.001)
        
        
        
        rval, f = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break   
    #network = Experiment()
            
#    for classes in os.listdir(imageDirectory):
#        for image in os.listdir(imageDirectory+"/"+classes):
#            imageNumber = imageNumber+1            
#            
#            img = DataUtil.prepareData(imageDirectory+"/"+classes+"/"+image, DataUtil.DATA_MODALITY["Face"], networkTopology[6][0][0][4]) 
#            #img = ImageProcessingUtil.whiten(img)
#            result = MCCNN.classify(network[len(network)-2],[img],trainingParameters[4])[0]*100
#            predicted = numpy.argmax(numpy.array(result), axis=0)
#            negative = numpy.array(result)[0]
#            neutral = numpy.array(result)[2]
#            positive = numpy.array(result)[1]
#            
##            if predicted == 0:
##                positive = 0
##                negative = numpy.array(result)[predicted]
##                neutral = 0
##            elif predicted == 1:
##                positive = 0
##                negative = 0
##                neutral =  numpy.array(result)[predicted]
##            else:
##                positive = numpy.array(result)[predicted]
##                negative = 0
##                neutral =  0
#                    
#            if len(plotNeutral) == 10:
#                del plotNeutral[0]
#                del plotNegative[0]
#                del plotPositive[0]
#                imageNumber = 10
#                plt.cla()
#                
#            plotNeutral.append(neutral)
#            plotNegative.append(negative)
#            plotPositive.append(positive)
#
#            
#            linePositive, = plt.plot(numpy.arange(imageNumber),plotPositive, label="Positive")
#            lineNegative, = plt.plot(numpy.arange(imageNumber), plotNegative, label="Negative")
#            lineNeutral, = plt.plot(numpy.arange(imageNumber), plotNeutral, label="Neutral")
#            
#            
#            plt.setp(linePositive, color='r', linewidth=2.0)
#            plt.setp(lineNegative, color='b', linewidth=2.0)
#            plt.setp(lineNeutral, color='y', linewidth=2.0)
#                                    
#            plt.legend([linePositive, lineNegative, lineNeutral], ["Positive", "Negative", "Neutral"])
#            plt.xlabel("Images")
#            plt.ylabel("Probability")
#            plt.title("Visualizing Faces")
#            plt.axis([0, imageNumber, -50, 150])
#            plt.yticks(numpy.arange(0,100, 5))
#   
#            #blue_patch = mpatches.Patch(color='blue', label='Test Set')
#            #plt.legend(handles=[red_patch])
#            
#            plt.draw()          
#            img = cv2.imread(imageDirectory+"/"+classes+"/"+image)
#            cv2.imshow("Face", img)
#            plt.pause(2)
#            
#            #print "Image:", imageDirectory+"/"+classes+"/"+image
#            print "Class:",     classes        
#            print "Class Predicted:", numpy.argmax(numpy.array(result), axis=0)
#            print "Result-Sum:", numpy.sum(numpy.array(result))
#            print "Result:", numpy.array(result)
            
            
            
#Experiment()
testJaffeSet()