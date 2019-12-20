import os

from Networks.Layers import InputLayer
from Networks.deconvlayer import DeconvLayer

import pylab
import theano
import theano.tensor as T
import numpy as np

from Utils import DataUtil, LogUtil

from Networks import Activations

DECONV_TYPES = {
    'max': 'max',
    'byFilter': 'byFilter'
}


def unravel(index, shape):
    res = []
    for i, val in enumerate(shape):
        x = np.prod(shape[i+1:], dtype=int)
        res.append((index//x) % val)
    return tuple(res)


def deconvolve(directory, deconv_layer, deconv_type=DECONV_TYPES['max'], vis_set='test',  num=1):
    network = DataUtil.loadNetworkState(directory)

    networkTopology = network[0]
    experimentParameters = network[2]
    networkState = network[4]

    outputLayer = networkTopology[5]
    categories = outputLayer[0]

    loadImagesStrategy = networkTopology[7]
    preLoadedFilters = networkTopology[8]

    baseDirectory = experimentParameters[0]
    experimentName = experimentParameters[2]

    channels = networkTopology[6]
    channels[0][0][5] = False

    experimentDirectory = baseDirectory + "/experiments/"+experimentName+"/"
    metricsDirectory = experimentDirectory+"/metrics/"

    log = LogUtil.LogUtil()
    log.createLog(experimentName, metricsDirectory)

    log.printMessage("--- Creating directory structure ---")
    for channel in range(len(channels)):
        cur_deconvlayer = deconv_layer[channel]

        visDirectory = experimentDirectory + '/' + 'visualisations_' + deconv_type + '_' + vis_set + 'Set/'
        layerDir = 'layer ' + str(cur_deconvlayer) + '/'
        channelDir = 'channel ' + str(channel) + '/'
        filtr_num = channels[channel][1][cur_deconvlayer - 1][0]
        for i in range(categories):
            for j in range(filtr_num):
                if not os.path.exists(visDirectory + channelDir
                                      + layerDir + 'filter '
                                      + str(j) + '/class ' + str(i) + '/'):
                    os.makedirs(visDirectory + channelDir
                                + layerDir + 'filter '
                                + str(j) + '/class ' + str(i) + '/')

    layers_by_channel = []
    #length_of_channel = []
    inputs = []
    output_by_channel = []
    pylab.gray()

    ## ######################
    ## ### Build Function ###
    ## ######################

    ###################
    ### Convolution ###
    ###################
    log.printMessage("--- Build Convolution part ---")

    for channel in range(len(channels)):
        cur_deconvlayer = deconv_layer[channel]
        log.printMessage(("--Channel:", channel))
        chan = []
        inputsPerChan = []
        inputStructure = channels[channel][0]
        layers = channels[channel][1]
        #length_of_channel.append(len(layers))

        imageSize = inputStructure[4]

        inputImagesPerChannel = inputStructure[0]
        imageStructure = inputStructure[2]
        #colorSpace = inputStructure[6]

        numberInputFirstLayer = 3

        x = T.matrix("x_" + str(channel))
        

        if(imageStructure
                == DataUtil.IMAGE_STRUCTURE["Static"]
                or imageStructure
                == DataUtil.IMAGE_STRUCTURE["StaticInSequence"]):
            x = x.reshape((1,
                           numberInputFirstLayer,
                           imageSize[0],
                           imageSize[1]))
            img = [1, 1, imageSize[0], imageSize[1]]
            inputImagesPerChannel = 1
        elif imageStructure == DataUtil.IMAGE_STRUCTURE["Sequence"]:
            x = x.reshape((1, inputImagesPerChannel,
                           numberInputFirstLayer, imageSize[0], imageSize[1]))
            img = [1, inputImagesPerChannel, imageSize[0], imageSize[1]]

        inputsPerChan.append(x)
        inputLayer = InputLayer(x, inputStructure, 1)

        output = inputLayer.output

        for layerParams, filters, layer_num in zip(channels[channel][1],
                                        networkState[0][channel],
                                        range(cur_deconvlayer)):

            print(layer_num)
            log.printMessage(("---Layer:", layer_num))
            layer = []
            activation = layerParams[6]
            usingInhibition = layers[layer_num][4]

            if activation == DataUtil.ACTIVATION_FUNCTION["Tanh"]:
                func = Activations.tanh
            elif activation == DataUtil.ACTIVATION_FUNCTION["ReLU"]:
                func = Activations.reLU

            W = filters[0][0]
            b = filters[0][1]
            flt = W.get_value().shape

            log.printMessage(("---- Filter Shape:", flt))
            log.printMessage(("---- Input Image Shape:", tuple(img)))

            if usingInhibition is not None:
                inhibitionFilters = preLoadedFilters[0][channel][layer_num][1]
                log.printMessage(("---- Using Inhibition:",
                                 usingInhibition is not None))
            else:
                inhibitionFilters = None

            deconv = DeconvLayer(
                    flt_shape=flt,
                    img_shape=tuple(img),
                    W=W,
                    bhid=b,
                    activationFunction=func,
                    inhibitionFilters=inhibitionFilters
                )

            output = deconv.get_hidden_values(output)

            layer.append(deconv)

            layer.append(layerParams[2])
            if(layerParams[2] and layer_num != (cur_deconvlayer-1)):
                output = deconv.max_pool(output)
                pool_size = layerParams[3]
                img[2] = img[2] / pool_size[0]
                img[3] = img[3] / pool_size[1]
                log.printMessage(("---- Max Pooling:", layerParams[2], "New Size:", tuple(img)))
            chan.append(layer)
            img[1] = flt[0]
        layers_by_channel.append(chan)

        output_by_channel.append(output)
        inputs.append(inputsPerChan)

    #####################
    ### Deconvolution ###
    #####################
    log.printMessage("--- Build Deconvolution part ---")

    for j, conv_output in enumerate(output_by_channel):
        cur_deconvlayer = deconv_layer[j]
        if deconv_type == DECONV_TYPES['max']:
            log.printMessage(" --- Deconvolve by choosing maximum activation")

            chan = []
            for i in range(flt[0]):
                cur_map = np.zeros((
                    img[0],
                    flt[0],
                    img[2],
                    img[3]
                    )).astype('f')
                ones = np.ones((
                    img[2]-(cur_deconvlayer*2),
                    img[3]-(cur_deconvlayer*2)
                    )).astype('f')
                ones = np.lib.pad(
                    ones, cur_deconvlayer,
                    mode='constant',
                    constant_values=0)

                cur_map[:, i, :, :] = ones
                max_img = conv_output * cur_map

                flt_map = []
                for k in range(num):
                    m, argm = T.max_and_argmax(max_img)
                    d = unravel(argm, img)

                    z = T.zeros_like(conv_output)
                    flt_map.append(T.set_subtensor(z[d], m))
                    max_img = T.set_subtensor(max_img[d], 0)

                chan.append(flt_map)
            output_by_channel[j] = chan
        elif deconv_type == DECONV_TYPES['byFilter']:
            log.printMessage(" --- Deconvolve by calculating the filter mean")

            chan = []
            x_coor = T.cast(T.scalar('x_coor'), 'int64')
            y_coor = T.cast(T.scalar('y_coor'), 'int64')
            inputs[j].append(x_coor)
            inputs[j].append(y_coor)

            for i in range(flt[0]):
                layer = []
                activated_neuron = T.zeros_like(conv_output)
                activated_neuron = T.set_subtensor(
                            activated_neuron[0, i, x_coor, y_coor],
                            conv_output[0, i, x_coor, y_coor])
                layer.append(activated_neuron)
                chan.append(layer)
            output_by_channel[j] = chan
        else:
            raise ValueError('Unknown parameter: deconv_type="%s"! Choose between "max" and "byFilter".' % deconv_type)
    channel = 0
    for conv_layers, feat_maps in zip(layers_by_channel, output_by_channel):
        cur_deconvlayer = deconv_layer[channel]
        conv_layers[cur_deconvlayer-1][1] = False

        for j, feature in enumerate(feat_maps):
            for i, val in enumerate(feature):
                for layer in reversed(conv_layers[:cur_deconvlayer]):
                    if layer[1]:
                        val = layer[0].unpool(val)

                    val = layer[0].get_reconstructed_input(val)
                feature[i] = val
        channel += 1

    log.printMessage("--- Creating Theano-Functions ---")
    log.printMessage(' --- This may take a while.')
    log.printMessage(' --- Warning! Using the "byFilter" option on large feature maps in large layers may lead to VERY long processing times!')
    log.printMessage(' --- Too many neurons to deconvolve. Each neuron would have its own deconvolution function.')
    log.printMessage(' --- For example:')
    log.printMessage(' --- Deconvolution of the fifth layer of a network, with 20 feature maps each with size 6x6')
    log.printMessage(' --- For this example the creation of the Theano-Functions takes about 10 minutes.')
    log.printMessage(' --- Usually only the last conv layer with the smallest feature maps should be used with this option')
    
    for j, feat_maps in enumerate(output_by_channel):
        for i, val in enumerate(feat_maps):
            deconv_func = theano.function(inputs[j], val)
            feat_maps[i] = deconv_func

    ## ######################
    ## ### Apply Function ###
    ## ######################
    log.printMessage("--- Apply Deconvolution ---")


    threshhold = 1
    if deconv_type == DECONV_TYPES['byFilter']:
        threshhold *= deconv_layer[0]

    pylab.gray()

    for channel, feat_maps in enumerate(output_by_channel):

        #channels[channel][0][3][0] = ("datasets/Jaffe_ternary/train")    # Set the percentage of
        #channels[channel][0][3][1] = ("datasets/Jaffe_ternary/valid")
        #channels[channel][0][3][2] = ("datasets/Jaffe_ternary/test")
        trainSet, validSet, testSet = DataUtil.loadData(log, baseDirectory, channels, loadImagesStrategy)

        if vis_set == 'train':
            log.printMessage(' --- Deconvolving the training set')
        elif vis_set == 'valid':
            log.printMessage(' --- Deconvolving the validation set')
            trainSet = validSet
        elif vis_set == 'test':
            log.printMessage(' --- Deconvolving the test set')
            trainSet = testSet
        else:
            raise ValueError('Unknown parameter: vis_set="%s"! Choose between "train", "valid" and "test".' % vis_set)

        n_train_batches = len(trainSet[0][0])

        cur_deconvlayer = deconv_layer[channel]
        layerDir = 'layer ' + str(cur_deconvlayer) + '/'

        channelDir = 'channel ' + str(channel) + '/'

        for index in range(n_train_batches):
            log.printMessage(("Image:", index+1, ' / ', n_train_batches))
            trainMiniBatch = DataUtil.loadImage(trainSet,
                                                channel,
                                                inputStructure[2],
                                                index,
                                                1,
                                                inputStructure[0],
                                                loadImagesStrategy,
                                                inputStructure[1],
                                                False)

            labels = trainMiniBatch[1]

            if inputStructure[2] == DataUtil.IMAGE_STRUCTURE["Sequence"]:
                examples = trainMiniBatch[0][0]
            else:
                examples = trainMiniBatch[0]

            for i, deconv_func in enumerate(feat_maps):
                feature_map_directory = 'filter ' + str(i) + '/class '
                result = []
                if deconv_type == DECONV_TYPES['byFilter']:
                    for u in range(0, img[2]):
                        for v in range(0, img[3]):
                            result += deconv_func(trainMiniBatch[0], u, v)
                else:
                    result = deconv_func(trainMiniBatch[0])
                frames = []

                if deconv_type == DECONV_TYPES['byFilter']:
                    mean = 0
                    for feature in result:
                        mean += feature
                    result = [mean]

                for feat_num, feature in enumerate(result):
                    for frame_num, val in enumerate(feature[0]):

                        if (val.sum() > (cur_deconvlayer*threshhold)
                                and np.amax(val) > 0.3):
                            frames.append(val)

                if len(frames) > 0:
                    for j, val in enumerate(examples):
                        pylab.axis('off')
                        pylab.imshow(val)
                        pylab.savefig(
                            visDirectory + channelDir
                            + layerDir
                            + feature_map_directory
                            + str(labels[0]) + '/'
                            + 'subj' + str(index) + '_frame' + str(j)
                        )
                    for j, val in enumerate(frames):
                        pylab.axis('off')
                        pylab.imshow(val)
                        pylab.savefig(
                            visDirectory + channelDir
                            + layerDir
                            + feature_map_directory
                            + str(labels[0]) + '/'
                            + 'subj' + str(index)
                            + '_max' + str(feat_num)
                            + '_frame' + str(j)
                            )

                pylab.cla()

    ## ################
    ## ### Clean Up ###
    ## ################

    log.printMessage("--- Cleaning up directory structure ---")
    log.printMessage(" --- Deleting directories of filters that did not produce any deconvolutions")
    for i in range(2):
        for root, dirs, files in os.walk(visDirectory):
            if not dirs:
                if not files:
                    log.printMessage("Deleting empty directory:")
                    log.printMessage(root)
                    os.rmdir(root)

#if __name__ == '__main__':
    ## Path to model.save file and the repetition to be visualised
    #expDir = [
              #['TEST_NEW_CODE', 6]
            #]
#
    ## Layers to be visualised
    ## -> channels
    ##    -> layer to be visualised for corresponding channel
    ## e.g [[2, 3]] layer 2 of channel 0 layer 3 of channel 3
    #layers = [[2]]
#
    #for i in expDir:
        #networkState = (
            #'/home/nimi/Documents/Experiments/experiments/'
            #+ i[0] + '/model/repetition_' + str(i[1]) + '_' + i[0] + '_.save'
            #)
        #for j in layers:
            #deconvolve(
                #networkState,
                #deconv_layer=j,
                #deconv_type='byFilter',
                #vis_set='test',
                #)
