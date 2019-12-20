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

import os



def Experiment():
                
    #Experiment parameters
    baseDirectory = "/data/pablo/"        
    
    experimentName = "Cohn-Kanade-feedback-WebCAM-SavingNetworks_4"    
    repetitions = 1     
    isGeneratingMetrics = True
    
    experimentParameters = []
    experimentParameters.append(baseDirectory)
    experimentParameters.append("-")  
    isGeneratingSynchronizedMetrics = False
    
    experimentParameters.append(experimentName) 
    experimentParameters.append(repetitions)
    experimentParameters.append(isGeneratingMetrics) 
    experimentParameters.append(isGeneratingSynchronizedMetrics) 
        
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
            
    
    layer01 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
    
    layer11 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
        
    layer2 = [20,(7,7), True, (2,2), None, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]            
    layer2i = [20,(7,7), True, (2,2), inhibitoryField0, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]            
    
    layer3 = [20,(5,5), True, (2,2), None, DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]            
    
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
     
    directoryDataStructure0 = ["Datasets/cohn-kanade-Feedback-4-frames", 60, 20, 20]
    
    inputStructure0 = [4, DataUtil.DATA_MODALITY["Face"], DataUtil.IMAGE_STRUCTURE["Sequence"], directoryDataStructure0, (64,49), False, DataUtil.COLOR_SPACE["GrayScale"], 0]        
    
    inputStructure1 = [1, DataUtil.DATA_MODALITY["Face"], DataUtil.IMAGE_STRUCTURE["StaticInSequence"], directoryDataStructure0, (64,49), False, DataUtil.COLOR_SPACE["GrayScale"], 3]    
        
    
    #inputStructure0 = [2, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Sequence"], "sequences_newCode", (50,50), False, DataUtil.COLOR_SPACE["GrayScale"], 0]    
    #inputStructure1 = [2, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["sequences_basic_emotions"], "sequences_newCode", (50,50), False, DataUtil.COLOR_SPACE["GrayScale"], 1]
    
    
    channel0 = (inputStructure0, [layer01, layer2], True)            
    channel1 = (inputStructure1, [layer11, layer2i ], True)       
      
     
    channels = [] 
     
    #channels.append(channel0)
    channels.append(channel1) 
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
        
    hiddenLayer1 = [250, DataUtil.ACTIVATION_FUNCTION["ReLU"], True, True, False, True, DataUtil.HIDDEN_LAYER_TYPE["Common"]]
    hiddenLayer2 = [50, DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False, True, DataUtil.HIDDEN_LAYER_TYPE["Common"]]
    
    
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
    
    isUsingMomentum = True
        
    maxTrainingEpochs = 50
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
    

def resize(image, size):            
        return numpy.array(cv2.resize(image,size))
        
def detectFace(img):     
    
        img2 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        
        cascade = cv2.CascadeClassifier("/informatik2/wtm/home/barros/Workspace/faceDetection/haarcascade_frontalface_alt.xml")
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
            
    
def testJaffeSet():
    
    #modelDirectory = "/export/home/hri/hriEnvironment/emotionRecognition/old_repetition_3_Cohn-Kanade_feedback_.save"
    #modelDirectory = "/data/pablo/experiments/Cohn-Kanade-feedback-Demo/model/repetition_3_Cohn-Kanade-feedback-Demo_.save"
    #modelDirectory =  "/data/pablo//experiments/Cohn-Kanade-feedback-WebCAM-SavingNetworks_3//model//repetition_0_BestTest_Cohn-Kanade-feedback-WebCAM-SavingNetworks_3_.save"
    modelDirectory =  "/data/pablo//experiments/Cohn-Kanade-feedback-WebCAM-SavingNetworks//model//repetition_0_BestTest_Cohn-Kanade-feedback-WebCAM-SavingNetworks_.save"
    
    networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectory)       
    
    experimentParameters[0] = os.path.dirname(os.path.abspath(__file__))
    experimentParameters.append(False)
        
    saveNetworkParameters = [False]
    
    #print "teste!"
    network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )    
        
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
        if frameNumber == 3:
            frameNumber = 0              
            rects = detectFace(f)          
            lookTo = []                                                     
            if not rects == None:                    
                for x1, y1, x2, y2 in rects:                        
                    img = f[y1:y2, x1:x2]                    
                img = resize(img,networkTopology[6][0][0][4])    
                faces.append(img)

                if len(faces) == 4:
                                         
                    result = MCCNN.classify(network[len(network)-2],[[faces[0],faces[1],faces[2],faces[3]],faces[2]],trainingParameters[4])[0]*100                
                    faces = []                
                    
                    predicted = numpy.argmax(numpy.array(result), axis=0)            
                    lookTo = ((x2 + x1)/2), ((y2+y1)/2)  
                    
                    if predicted == 0:
                            color = (192, 19, 19)
                            text = "Negative"                                                        
        
                             
                    elif predicted == 2:
                            color = (200, 200, 20)                                        
                            text = "Neutral"                    
        
                    else:
                            color = (19, 71, 192)
                            text = "Positive"       
                    recognized = True
        
        
            else:
                recognized = False
                                                 
        if recognized:        
            cv2.circle(f, lookTo, 10, color, thickness=-1)                                                                   
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)        
            cv2.rectangle(f, (x1- abs(x1-x2), y1-int(abs(y1-y2)/4)), (x2+abs(x1-x2), y2+abs(y1-y2)*2), color, 2)                
            cv2.putText(f,text, (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color)                                                        
            
        cv2.imshow("FaceLiveImage", f)                 
        rval, f = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break       
             
            
            
Experiment()
#testJaffeSet()  
