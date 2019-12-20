# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from Utils import DataUtil
import MCCNNExperiments


def Experiment():
                
    #Experiment parameters
    baseDirectory = "/data/datasets/mini-genres/"        
    
    experimentName = "Experiment_Mini-Genres_All_01"    
    repetitions = 1    
    isGeneratingMetrics = True
    isGeneratingSynchronizedMetrics = True
    
    experimentParameters = []
    experimentParameters.append(baseDirectory)
    experimentParameters.append("-")  
    
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

    
    layerAudio = [4,(5,5), None, (2,2), None , DataUtil.LAYER_TYPE["AudioFilters"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]           
#    layer0 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
#    layer1 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
#    layer2 = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layerA0 = [1024,(64,5), None, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]
        
    layerA = [5,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
    layerB = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True] 
    layerC = [20,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True] 

     
    layerEx13 = [1,(1,3), None, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]    
    layerEx53 = [5,(1,3), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
    
    layerEx15 = [1,(1,5), None, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]    
    layerEx55 = [5,(1,5), None, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]    
    
    layerEx13p = [1,(1,3), True, (1,5), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]    
    layerEx53p = [10,(1,3), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
    layerEx532p = [20,(1,3), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["ReLU"], True, False, True]        
    
    layerEx15p = [1,(1,5), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]    
    layerEx55p = [5,(1,5), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    
     
     
    layer0 = [5,(1,11), None, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]            
    layer1 = [5,(1,5), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer2 = [5,(1,5), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer3 = [5,(1,3), True, (1,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]     
    
    layer0b = [5,(1,11), None, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]            
    layer1b = [5,(1,5), True, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer2b = [5,(1,5), True, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer3b = [5,(1,3), True, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    
    layer0a = [5,(11,1), True, (2,1), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer1a = [5,(5,1), True, (2,1), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]            
    layer2a = [5,(5,1), True, (2,1), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    layer3a = [5,(5,1), True, (2,1), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    
    #layer0b = [10,(5,5), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]        
    #layer1b = [20,(3,3), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]            
    #layer2b = [40,(5,1), True, (2,1), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True] 
    
    
    #layer2 = [10,(3,3), True, (2,2), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], True, False, True]                  
    #layer3 = [10,(1,5), True, (1,3), None , DataUtil.LAYER_TYPE["Common"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True]                  

    
    #layer2 = [1,(3,3), False, (2,2), inhibitoryField0, DataUtil.LAYER_TYPE["SobelX"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False]            
    
    #layer3 = [1,(3,3), False, (2,2), None, DataUtil.LAYER_TYPE["SobelY"], DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, False]            
        

    """     
    Channels Topology
    The channel topology is composed of two parameters: (colorSpace, imageStructue)
    inputStructure  - represents which inputStructure this channel uses.        
    Layers - Layers that compose this channel    
    bool - attachThisChannel - Indicate if this channel is attached to the hidden layer (Use False when the channel is used as input to a crossChannel)                
    """ 
    
    """
    Directory structure for each channel
    String -  name of the folder with the testing set
    int - percent of the data separated for training
    int- percent of the data separated for validation
    int- percent of the data separated for testing
    
    or
    
    
    String - name of the folder with the training data
    String- name of the folder with the validation data
    String - name of the folder with the testing data 
    """ 
    directoryDataStructure0 = ["/audio_GTZAN/train", "/audio_GTZAN/validation", "/audio_GTZAN/test"] 
    #directoryDataStructure0 = ["AudioPerSubject/not_separated/DC", 60,20,20] 
    directoryDataStructure0a = ["All_MFCC_11/1024/training/", 60, 20, 20] 
    #directoryDataStructure0a = ["Train_Audio", 60, 20, 20] 
    #directoryDataStructure1 = ["MFCC", 60, 20, 20] 
    
    #directoryDataStructure0 = [("AudioPerSubject/DC/train"), ("AudioPerSubject/DC/validation"),("AudioPerSubject/DC/test")]
    #directoryDataStructure0a = [("DC_JE_MFCC_1/1024/train"), ("DC_JE_MFCC_1/1024/validation"),("DC_JE_MFCC_1/1024/test")]
    
    directoryDataStructure0b = [("audio_GTZAN_Separated_delta_delta_MFCC_Small_Sequences_10/train"), ("audio_GTZAN_Separated_delta_delta_MFCC_Small_Sequences_10/validation"),("audio_GTZAN_Separated_delta_delta_MFCC_Small_Sequences_10/test")]
    #directoryDataStructure0 = [("audio_GTZAN_Separated_Spectrum_Small_Sequences_10/train"), ("audio_GTZAN_Separated_Spectrum_Small_Sequences_10/validation"),("audio_GTZAN_Separated_Spectrum_Small_Sequences_10/test")]
    #directoryDataStructure0a = [("audio_GTZAN_Separated_MFCC_Small_Sequences_10/train"), ("audio_GTZAN_Separated_MFCC_Small_Sequences_10/validation"),("audio_GTZAN_Separated_MFCC_Small_Sequences_10/test")]
    directoryDataStructure1 = [("audio_GTZAN_80Percent/train"), ("audio_GTZAN_80Percent/validation"),("audio_GTZAN_80Percent/test")]
    directoryDataStructure2 = [("audio_GTZAN_80Percent/train"), ("audio_GTZAN_80Percent/validation"),("audio_GTZAN_80Percent/test")]
    #directoryDataStructure0 = [("sequences_DEMO"), ("sequences_DEMO"),("sequences_DEMO")]
    
    
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
         
    inputStructure0 = [1, DataUtil.DATA_MODALITY["Audio"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure0, (71,26), False, DataUtil.COLOR_SPACE["GrayScale"], 1]        
    inputStructure0a = [1, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure0a, (35,13), False, DataUtil.COLOR_SPACE["GrayScale"], 1]        
    inputStructure0b = [1, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure0b, (41,13), False, DataUtil.COLOR_SPACE["GrayScale"], 1]        
    inputStructure1 = [1, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure1, (151,13), False, DataUtil.COLOR_SPACE["GrayScale"], 1]        
    inputStructure2 = [1, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure2, (151,13), False, DataUtil.COLOR_SPACE["GrayScale"], 1]        
    
    
    channel0 = (inputStructure0, [layerEx53, layerEx53p, layerEx532p], True)            
    #channel0 = (inputStructure0, [layerEx53, layerEx53p, layerEx532p], True)    
    channel0a = (inputStructure0a, [layerEx53, layerEx53p, layerEx532p], True)   
    channel0b = (inputStructure0b, [layer0, layer1, layer2], True)   
    #channel1 = (inputStructure1, [layer0, layer1, layer2], True)            
    #channel2 = (inputStructure1, [layer0, layer1, layer2], True)            
    
                  
    channels = [] 
     
    channels.append(channel0)
    #channels.append(channel0a) 
    #channels.append(channel0b) 
    #channels.append(channel2)
    #channels.append(channel3)
 

  
    """ Cross Channels
    The CrossChannels topology is composed of two parameters: (inputChannels, layers)
    (int,int[])  - represents which layer from which channels will serve as input to the crossChannelLayers
             
    Layers - Layers that compose this Crosschannel
    
    The input channels must have the same shape(the same image size)
    """ 
      
    
    crossConvolutionChannels = []
#    
    crossConvChannel1 = [ [(0,1), (1,1)], [layer1]] 
#    
#    crossConvChannel2 = [ [(0,0), (1,0)], [layer3]]
    
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
    (HIDDEN_LAYER_TYPE) - defines the type of hidden layer to be used
    
    """ 
        
    hiddenLayer1 = [500, DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, False, True, DataUtil.HIDDEN_LAYER_TYPE["Common"]]
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
        
    outputLayer = [5, True, False]

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
        
    maxTrainingEpochs = 1000
    L1Regularization = 0.001
    L2Regularization = 0.0001
    
    """
    Training Parameters 
    """
    batchSize = 10
    inititalLearningrate = 0.01
    momentum = 0.99
      
    """
    Training data parameters 
    (int, int, int) dataSetDivision = porcentage of data for training, validation and testing respectively.
    """    
    
    trainingParameters = []
    trainingParameters.append(isTraining)    
    trainingParameters.append("") 
    trainingParameters.append(isUsingMomentum)
    trainingParameters.append(maxTrainingEpochs)
    trainingParameters.append(batchSize)
    trainingParameters.append(inititalLearningrate)
    trainingParameters.append(momentum)
    trainingParameters.append("-")
    trainingParameters.append("")
    trainingParameters.append(L1Regularization) 
    trainingParameters.append(L2Regularization)
    trainingParameters.append("")
    trainingParameters.append("")
        
    #Visualization parameters
    
    isCreatingHintongDiagrams = False #""" TO DO """
    isCreatingOutputImages    = False #""" TO DO """
    isCreatingConvFeatures = False #""" TO DO """
    isVisualizingTrainingEpoches = False
    """ 
    (bool, bool, bool, bool) isVisualizingTrain - The first parameter is related to visualize the filters change during training, the second one is to save the filters
    change as an image. The third one creates a live visualization of the filters during training. The forth one creates video from the filters updates.
    """
    visualizeFiltersTraining = False 
    saveFilters= False
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

    
    MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )
    
    
Experiment()


# -*- coding: utf-8 -*-

