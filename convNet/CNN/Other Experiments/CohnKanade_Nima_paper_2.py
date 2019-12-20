# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from Utils import DataUtil
import MCCNNExperiments


def Experiment():

    #Experiment parameters
    baseDirectory = "/data/barros/"

    experimentName = "New_Test"
    repetitions = 1
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
    inhibitoryField0 = (DataUtil.LAYER_TYPE["Common"], True)

    layer1 = [10, (5, 5), True, (2, 2), None, DataUtil.LAYER_TYPE["Common"],
              DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, True]
    layer2 = [10, (5, 5), True, (2, 2), None, DataUtil.LAYER_TYPE["Common"],
              DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, True]
    layer3 = [10, (5, 5), True, (2, 2), None, DataUtil.LAYER_TYPE["Common"],
              DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, True]
    layer4 = [10, (5, 5), True, (2, 2), None, DataUtil.LAYER_TYPE["Common"],
              DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, True]
    layer5 = [10, (5, 5), True, (2, 2), None, DataUtil.LAYER_TYPE["Common"],
              DataUtil.ACTIVATION_FUNCTION["ReLU"], False, True, True]

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
    #directoryDataStructure0 = ["images3", 50, 25, 25]
    #directoryDataStructure0 = [("HRI_dataset_reduced_resized_grayscale/train"), ("HRI_dataset_reduced_resized_grayscale/valid"),("HRI_dataset_reduced_resized_grayscale/test")]
    directoryDataStructure0 = [("images/train"), ("images/valid"), ("images/test")]

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

    inputStructure0 = [4, DataUtil.DATA_MODALITY["Face"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure0, (256, 256), False, DataUtil.COLOR_SPACE["Whiten"], 0]
    #inputStructure0 = [1, DataUtil.DATA_MODALITY["Image"], DataUtil.IMAGE_STRUCTURE["Static"], directoryDataStructure0, (64,64), False, DataUtil.COLOR_SPACE["Whiten"], 2]

    channel0 = (inputStructure0,
                [layer1, layer2, layer3, layer4, layer5],
                True)

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
#
#    crossConvChannel1 = [ [(0,1), (1,1)], [layer3]]
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

    hiddenLayer1 = [1000, DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True, True,
                    DataUtil.HIDDEN_LAYER_TYPE["Common"]]
    hiddenLayer2 = [500, DataUtil.ACTIVATION_FUNCTION["Tanh"], False, False, True, True,
                    DataUtil.HIDDEN_LAYER_TYPE["Common"]]

    hiddenLayers = []
    hiddenLayers.append(hiddenLayer1)
    hiddenLayers.append(hiddenLayer2)

    """
    Each output layer is composed of 3 parameters: (numberOfOutputUnits, L1Reg, L2Reg)

    int numberOfOutputUnits - Number of output Units in this layer.
    (bool) L1Regularization - defines if the layer is present on the L1Regularization strategy
    (bool) L2Regularization - defines if the layer is present on the L2Regularization strategy

    """

    outputLayer = [3, False, True]

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

    maxTrainingEpochs = 100
    L1Regularization = 0.001
    L2Regularization = 0.0001

    """
    Training Parameters
    """
    batchSize = 50
    inititalLearningrate = 0.1
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


    MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )


Experiment()
