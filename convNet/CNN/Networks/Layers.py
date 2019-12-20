# -*- coding: utf-8 -*-
import theano.tensor as T
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet.conv import conv2d
from theano.tensor.extra_ops import repeat


import Activations

import theano.sparse.sandbox.sp

import numpy
import cv2

from Utils import DataUtil
from Utils.ImageProcessingUtil import grayImageTheanoFunction, normalizeInputTheanoFunction, whitenImage


rng = numpy.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))



class InputLayer(object):
    def __init__(self, inputImage, inputStructure, batchSize):
                        
        inputImageSize =inputStructure[4]
        inputImages = inputStructure[0]
        channelColorSpace = inputStructure[6]
        imageStructure = inputStructure[2]                    
        dataModality = inputStructure[1]    
        
#        if dataModality == DataUtil.DATA_MODALITY["Audio"]:
#            a = grayImageTheanoFunction(inputImage)       
#            if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"]  : 
#                #a = inputImage
#                #a = grayImageTheanoFunction(inputImage)       
#                a = a.dimshuffle(0,'x',1,2)
#                outputShape = (batchSize, 1, inputImageSize[1],inputImageSize[0])
#            else:
#                outputShape = (batchSize, inputImages, inputImageSize[1],inputImageSize[0])
#            
#            channelsOutput = a    
#            channelsOutput = normalizeInputTheanoFunction(a) 
#            
#            outputShape = outputShape
#            
#        elif channelColorSpace == DataUtil.FEATURE_TYPE["GrayScale"]:
        if channelColorSpace == DataUtil.FEATURE_TYPE["GrayScale"] or dataModality == DataUtil.DATA_MODALITY["Audio"] :
            a = grayImageTheanoFunction(inputImage)       
            if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"]  : 
                a = a.dimshuffle(0,'x',1,2)                                
                outputShape = (batchSize, 1, inputImageSize[1],inputImageSize[0])
            else:
                outputShape = (batchSize, inputImages, inputImageSize[1],inputImageSize[0])
                
            channelsOutput = normalizeInputTheanoFunction(a) 
            
            outputShape = outputShape
            
         
        elif channelColorSpace == DataUtil.FEATURE_TYPE["Whiten"]:
            a = grayImageTheanoFunction(inputImage)                   
            
            if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"] :                     
                a = a.dimshuffle(0,'x',1,2)
                outputShape = (batchSize, 1, (inputImageSize[1]),(inputImageSize[0]))
            else:
                outputShape = (batchSize, inputImages, (inputImageSize[1]),(inputImageSize[0]))
            
            a = whitenImage(a)
            channelsOutput = normalizeInputTheanoFunction(a)                
            outputShape = outputShape
            
            
        elif channelColorSpace == DataUtil.FEATURE_TYPE["RGB"]:
            if imageStructure == DataUtil.IMAGE_STRUCTURE["Static"] or imageStructure == DataUtil.IMAGE_STRUCTURE["StaticInSequence"] :                    
                a = inputImage.dimshuffle(0,3,1,2)
                outputShape = (batchSize, 3, inputImageSize[1],inputImageSize[0])
            elif imageStructure == DataUtil.IMAGE_STRUCTURE["Sequence"]:
                a = inputImage.dimshuffle(0,1,4,2,3)
                outputShape = (batchSize,inputImages,3, inputImageSize[1],inputImageSize[0])

            channelsOutput = normalizeInputTheanoFunction(a) 
            outputShape = outputShape
                        
        
        self.outputShape =  outputShape
        self.input = inputImage
        self.output = channelsOutput            
#        print "Output Shape:", outputShape
        
                
                
            
class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self,activationFunction, filterType, useInhibition, usePooling, rng, filters, inhibitionFilters, input, filter_shape, image_shape, poolsize):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type useInhibition: Boolean
        :param useInhibition: Defines if this layer uses shunting inhibition or not.
        
        :type useInhibition: Boolean
        :param useInhibition: Defines if this layer uses pooling or not.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type universalFeatures: Boolean
        :param universalFeatures: Defines if this layer uses CAE filters or not.
        
        :type layerOrder: Int
        :param layerOrder: Indicates which is the order of this layer when the network is saved.
        
        :type loadFrom: String
        :param layerOrder: Indicates where the saved network is storaged.
        
        :type parametersToLoad: String
        :param layerOrder: Indicates where the CAE matrix is storaged.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
         
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.useInhibition = useInhibition
        self.activationFunction = activationFunction
        self.filterType = filterType
        self.usePooling = usePooling
        self.rng = rng
        self.filters = filters
        self.inhibitionFilters = inhibitionFilters
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize

#        print "Weights:", self.filters[0].astype(theano.config.floatX).eval()
#        print "Bias:", self.filters[1].astype(theano.config.floatX).eval()
#        raw_input("here")
        
        if not(self.filters == None):
            
#            print "Shape0:",  numpy.array(self.filters[0]).shape
#            print "Shape1:",  numpy.array(self.filters[1]).shape
#            
#            print "Filters 0:", self.filters[0]
#            print "Filters 1:", self.filters[1]
            print  "loading filters...."
            self.W = self.filters[0].eval()
            self.b = self.filters[1].eval()

            self.W = theano.shared(self.W,
                                  borrow=True)
            self.b = theano.shared(value=self.b, borrow=True)
            
        else:    
            Ww, filter_shape = getFilters(self.filterType, filter_shape)                            
            
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)        
                  
        
            self.W = theano.shared(Ww,
                                  borrow=True)   
            
            self.b = theano.shared(value=b_values, borrow=True)
        
    def getOutput(self):
        if self.useInhibition!= None:
                                         
            self.decayTerm = 0.5
            
            #print "DecayTerm:",dTerm
            
            if not(self.inhibitionFilters == None):            
                self.wInhibitory = self.inhibitionFilters[0].astype(theano.config.floatX)  
                self.bInhibition = self.inhibitionFilters[1].astype(theano.config.floatX)  
                self.decayTerm = self.inhibitionFilters[2].astype(theano.config.floatX)
                
                inhibition_filter_shape = self.filter_shape
            else:
                filtersInhibition, inhibition_filter_shape = getFilters(self.useInhibition[0], self.filter_shape)
                self.bInhibition = numpy.zeros((inhibition_filter_shape[0],), dtype=theano.config.floatX)  
            
                self.bInhibition = theano.shared(value=self.bInhibition, borrow=True)
                self.wInhibitory = theano.shared(filtersInhibition,borrow=True)                
                #self.decayTerm = theano.shared(dTerm,borrow=True)                
           # newW = (self.W / (self.decayTerm + self.wInhibitory)).astype(theano.config.floatX) 
            
            if len(self.image_shape) == 4:
                #output = conv.conv2d(input=self.input, filters= newW , filter_shape=inhibition_filter_shape, image_shape=self.image_shape)
                activation = conv.conv2d(input=self.input, filters= self.W , filter_shape=self.filter_shape, image_shape=self.image_shape, border_mode="valid")
                inhibitoryActivation = conv.conv2d(input=self.input, filters= self.wInhibitory , filter_shape=inhibition_filter_shape, image_shape=self.image_shape)
            else:
                #output = T.nnet.conv3d2d.conv3d(signals=self.input, filters=newW,filters_shape=inhibition_filter_shape, signals_shape=self.image_shape,border_mode='valid')    
                
                activation = T.nnet.conv3d2d.conv3d(signals=self.input, filters=self.W,filters_shape=self.filter_shape, signals_shape=self.image_shape,border_mode='valid')    
                inhibitoryActivation = T.nnet.conv3d2d.conv3d(signals=self.input, filters=self.wInhibitory,filters_shape=inhibition_filter_shape, signals_shape=self.image_shape,border_mode='valid')    
                
            
            activation = Activations.applyActivationFunction(self.activationFunction,activation + self.b.dimshuffle('x', 0, 'x', 'x') )          
            inhibitoryActivation = Activations.applyActivationFunction(self.activationFunction,inhibitoryActivation + self.bInhibition.dimshuffle('x', 0, 'x', 'x') )          
            
            output = (activation / (self.decayTerm +inhibitoryActivation)).astype(theano.config.floatX)  
            #output = inhibitoryActivation
            
            self.paramsInhibition = [self.wInhibitory, self.bInhibition]    
                                  
        else:            
            if len(self.image_shape) == 4:
                
                
#                sparseMatrix = theano.sparse.sandbox.sp.ConvolutionIndices.evaluate(inshp = (1,35,13),  kshp=(1,5), nkern=5)
#                
#                
#                
#                print "Shape:", numpy.array(sparseMatrix).shape
#                
#                print sparseMatrix
#                
#                imagePatches = T.dot(sparseMatrix,self.input)
#                
#                output = T.dot(imagePatches,self.W)                
                
                output = conv.conv2d(input=self.input, filters= self.W , filter_shape=self.filter_shape, image_shape=self.image_shape, border_mode="valid")
            else:
                output = T.nnet.conv3d2d.conv3d(signals=self.input, filters=self.W,filters_shape=self.filter_shape, signals_shape=self.image_shape,border_mode='valid')    
                
                #print "Input Shape:", self.image_shape
                #print "Filter shape:", self.filter_shape
                
            output = Activations.applyActivationFunction(self.activationFunction,output + self.b.dimshuffle('x', 0, 'x', 'x') )   
        
        if  len(self.image_shape) == 4:               
            inputImageSizeX = (self.image_shape[2] - self.filter_shape[2] +1) 
            inputImageSizeY = (self.image_shape[3] - self.filter_shape[3] +1)
            
            #inputImageSizeX = (self.image_shape[2] + self.filter_shape[2] -1) 
            #inputImageSizeY = (self.image_shape[3] + self.filter_shape[3] -1)
        else:
            inputImageSizeX = (self.image_shape[3] - self.filter_shape[3] +1) 
            inputImageSizeY = (self.image_shape[4] - self.filter_shape[4] +1)
            
            #inputImageSizeX = (self.image_shape[3] + self.filter_shape[3] -1) 
            #inputImageSizeY = (self.image_shape[4] + self.filter_shape[4] -1)
        
        if self.usePooling:    
            output = downsample.max_pool_2d(input=output,ds=self.poolsize, ignore_border=True)
            inputImageSizeX = inputImageSizeX / self.poolsize[0]
            inputImageSizeY = inputImageSizeY / self.poolsize[1]
        
        #output = normalizeInputTheanoFunction(output)                              
             
                      
        if  len(self.image_shape) == 5:
        #    print "Reshape"
            self.output =  output.reshape((self.image_shape[0], self.filter_shape[0],inputImageSizeX,inputImageSizeY ))
        else:
            self.output = output
    
        
        
        self.outputShape =  (self.image_shape[0], self.filter_shape[0],inputImageSizeX,inputImageSizeY )                
        
        print "OutputShape:", self.outputShape
       # print "Conv layer OutputShape:", (image_shape[0], filter_shape[0],inputImageSizeX,inputImageSizeY ) 
       
        self.params = [self.W, self.b]            
        
                
#        print "Input Shape:", self.image_shape
#        print "Filter shape:", self.filter_shape
#        print "Output shape:", self.filter_shape
 

def rescale_weights(params, incoming_max):
    incoming_max = numpy.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * numpy.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)
        

def getFilters(layerType, filter_shape):
    
    if layerType == DataUtil.LAYER_TYPE["Common"]:          
        
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))   # /POOLSIZE          
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))        
        
        rng = numpy.random.RandomState(None)        
        
        return numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=theano.config.floatX), filter_shape
                
    elif layerType == DataUtil.LAYER_TYPE["SobelX"]: 
        filter_shape = (filter_shape[0], filter_shape[1], 3,3)
        return getSobelFilter("x",filter_shape ), filter_shape
        
        
    elif layerType == DataUtil.LAYER_TYPE["SobelY"]: 
        filter_shape = (filter_shape[0], filter_shape[1], 3,3)
        return getSobelFilter("y",filter_shape ), filter_shape        
        
        
    elif layerType == DataUtil.LAYER_TYPE["Gabor"]: 
        loadedParams = DataUtil.loadState(DataUtil.GABOR_FILTERS_DIRECTORY,1)                                
        Ww = loadedParams[0][0].astype(theano.config.floatX)        
        filter_shape = (20, filter_shape[1], 11,11)
    
        return Ww, filter_shape
        
    elif layerType == DataUtil.LAYER_TYPE["AudioFilters"]: 
        filter_shape = (filter_shape[0], filter_shape[1], 5,5)
        return getAudioFilters(filter_shape ), filter_shape
        


def getAudioFilters(filter_shape):
    Ww = numpy.asarray(rng.uniform(low=-0, high=0, size=filter_shape),dtype=theano.config.floatX)
    
#    print "Shape:", numpy.array(Ww).shape
    for i in range(len(Ww)):    
        
            Ww[0][0][0] = [0, 0, 1, 0, 0]
            Ww[0][0][1] = [0, 0, 1, 0, 0]
            Ww[0][0][2] = [0, 0, 1, 0, 0]
            Ww[0][0][3] = [0, 0, 1, 0, 0]
            Ww[0][0][4] = [0, 0, 1, 0, 0]                        
            
            Ww[1][0][0] = [1, 0, 0, 0, 0]
            Ww[1][0][1] = [0, 1, 0, 0, 0]
            Ww[1][0][2] = [0, 0, 1, 0, 0]
            Ww[1][0][3] = [0, 0, 0, 1, 0]
            Ww[1][0][4] = [0, 0, 0, 0, 1]
            
            Ww[2][0][0] = [0, 0, 0, 0, 0]
            Ww[2][0][1] = [0, 0, 0, 0, 0]
            Ww[2][0][2] = [1, 1, 1, 1, 1]
            Ww[2][0][3] = [0, 0, 0, 0, 0]
            Ww[2][0][4] = [0, 0, 0, 0, 0]
            
            Ww[3][0][0] = [0, 0, 0, 0, 1]
            Ww[3][0][1] = [0, 0, 0, 1, 0]
            Ww[3][0][2] = [0, 0, 1, 0, 0]
            Ww[3][0][3] = [0, 1, 0, 0, 0]
            Ww[3][0][4] = [1, 0, 0, 0, 0]            
    
    return Ww
    

def getSobelFilter(direction, filter_shape):

    Ww = numpy.asarray(rng.uniform(low=-0, high=0, size=filter_shape),dtype=theano.config.floatX)
             
    if(direction == "x"):
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        Ww[i][u][0] = [1,0,-1]
                        Ww[i][u][1] = [2,0,-2]
                        Ww[i][u][2] = [1,0,-1]
    else:
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        Ww[i][u][0] = [1,2,1]
                        Ww[i][u][1] = [0,0,0]
                        Ww[i][u][2] = [-1,-2,-1]   
                        
    return Ww


def drop(input, p=0.5, rng=rng): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
    """            
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask
            

    
def init_W_b(W, b, rng, n_in, n_out):
    
    # for a discussion of the initialization, see   
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa 
    if W is None:    
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6./(n_in + n_out)),
                high=numpy.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function 
    if b is None:
        b_values = numpy.ones((n_out,), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b
    
    
class DropoutHiddenLayer(object):
    def __init__(self, activationFunction, rng, is_train, filters, input, n_in, n_out, W=None, b=None,
                 p=0.5):
                     
        
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_emodelOutputxamples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input
        # end-snippet-1
            
        if filters is None:                
            self.W, self.b  = init_W_b(W, b, rng, n_in, n_out)
#            self.W = theano.shared(value=W,
#                                name='W', borrow=True)
#            self.b = theano.shared(value=b,
#                               name='b', borrow=True)
        else: 
            self.W = filters[0].eval()
            self.b = filters[1].eval()            
            self.W  = theano.shared(value=self.W , name='W', borrow=True)
            self.b  = theano.shared(value=self.b, name='b', borrow=True)
            print  "loading filters...."
        
        

        lin_output = T.dot(input, self.W) + self.b
        
        output = Activations.applyActivationFunction(activationFunction,lin_output )                
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(numpy.cast[theano.config.floatX](1./p) * output)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, output)
        
        # parameters of the model
        self.params = [self.W, self.b]
 

class HiddenLayer(object):
    def __init__(self, activationFunction ,rng, filters ,input, n_in, n_out, W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        filters = None
        if filters is None:           
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)   
            
            self.W = theano.shared(value=W_values,name='W', borrow=True)
            
            self.b = theano.shared(value=b_values,name='b', borrow=True)
        else:
            self.W = filters[0].eval()
            self.b = filters[1].eval()    
            
            self.W = theano.shared(value=self.W,name='W', borrow=True)
            self.b = theano.shared(value=self.b,name='b', borrow=True)
            print  "loading filters...."              

        lin_output = T.dot(input, self.W) + self.b
        
        self.output = Activations.applyActivationFunction(activationFunction,lin_output) 
                
        # parameters of the model
        self.params = [self.W, self.b]
        


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, filters, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        #filters = None
        
        #raw_input("here1")
        #print filters
        if filters is None:        
            W_values = numpy.zeros((n_in, n_out),dtype=theano.config.floatX)
            b_values = numpy.zeros(n_out,dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values,name='W', borrow=True)
            self.b = theano.shared(value=b_values,name='b', borrow=True)
        else:
            self.W = filters[0].eval()
            self.b = filters[1].eval()
            #print "filters:", numpy.sum(self.W)
            #raw_input("a")												
            self.W = theano.shared(value=self.W,name='W', borrow=True)
            self.b = theano.shared(value=self.b,name='b', borrow=True)            
            print  "loading filters...."

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b] 

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example andCHANNEL_TYPE
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type)) 
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LSTMLayer(object):    
        
            
	def __init__(self, input, n_in, n_hidden, outputNeurons, convLayersOutput, activationFunction, channelLayers, channels, filters):
		
		# type definition
		self.dtype=theano.config.floatX
		
		# squashing of the gates should result in values between 0 and 1, therefore we use the logistic function
		self.sigma = lambda x: 1 / (1 + T.exp(-x))
		
		# for the other activation function we use the tanh
		self.act = T.tanh
		
		#input matrix
		#self.v = T.matrix(dtype=self.dtype)
		
		#Layer Parameters, used for weight matrices definition
		self.n_in=n_in
		self.n_hidden = n_hidden
		self.n_i = self.n_c = self.n_o = self.n_f = self.n_hidden		
		           
		self.input = input
		self.activationFunction = activationFunction
		self.outputNeurons = outputNeurons
           
		if not(filters == None):
                    self.W_xi = self.filters[0].astype(theano.config.floatX) 
                    self.W_hi = self.filters[1].astype(theano.config.floatX) 
                    self.W_ci = self.filters[2].astype(theano.config.floatX)  
                    self.b_i = self.filters[3].astype(theano.config.floatX)
                    self.W_xf = self.filters[4].astype(theano.config.floatX)                    
                    self.W_hf = self.filters[5].astype(theano.config.floatX)                                        
                    self.W_cf = self.filters[6].astype(theano.config.floatX)
                    self.b_f = self.filters[7].astype(theano.config.floatX)
                    self.W_xc = self.filters[8].astype(theano.config.floatX)
                    self.W_hc = self.filters[9].astype(theano.config.floatX)
                    self.b_c = self.filters[10].astype(theano.config.floatX)
                    self.W_xo = self.filters[11].astype(theano.config.floatX)
                    self.W_ho = self.filters[12].astype(theano.config.floatX)
                    self.W_co = self.filters[13].astype(theano.config.floatX)
                    self.b_o = self.filters[14].astype(theano.config.floatX)                                     
                    self.c0 = self.filters[15].astype(theano.config.floatX)
                    self.W_hy = self.filters[16].astype(theano.config.floatX)
                    self.b_y = self.filters[17].astype(theano.config.floatX)
                    
		else:         
        		# Weight initialisation
        		self.W_xi = theano.shared(self.sample_weights(self.n_in, self.n_i))  
        		self.W_hi = theano.shared(self.sample_weights(self.n_hidden, self.n_i))  
        		self.W_ci = theano.shared(self.sample_weights(self.n_c, self.n_i))  
        		self.b_i = theano.shared(numpy.cast[self.dtype](numpy.random.uniform(-0.5,.5,size = self.n_i)))
        		self.W_xf = theano.shared(self.sample_weights(self.n_in, self.n_f)) 
        		self.W_hf = theano.shared(self.sample_weights(self.n_hidden, self.n_f))
        		self.W_cf = theano.shared(self.sample_weights(self.n_c, self.n_f))
        		self.b_f = theano.shared(numpy.cast[self.dtype](numpy.random.uniform(0, 1.,size = self.n_f)))
        		self.W_xc = theano.shared(self.sample_weights(self.n_in, self.n_c))  
        		self.W_hc = theano.shared(self.sample_weights(self.n_hidden, self.n_c))
        		self.b_c = theano.shared(numpy.zeros(self.n_c, dtype=self.dtype))
        		self.W_xo = theano.shared(self.sample_weights(self.n_in, self.n_o))
        		self.W_ho = theano.shared(self.sample_weights(self.n_hidden, self.n_o))
        		self.W_co = theano.shared(self.sample_weights(self.n_c, self.n_o))
        		self.b_o = theano.shared(numpy.cast[self.dtype](numpy.random.uniform(-0.5,.5,size = self.n_o)))
        		self.W_hy = theano.shared(self.sample_weights(self.n_hidden, self.outputNeurons))
        		self.b_y = theano.shared(numpy.zeros(self.outputNeurons, dtype=self.dtype))          
                
        		# initialisation of cell state 
        		self.c0 = theano.shared(numpy.zeros(self.n_hidden, dtype=self.dtype))
           #initialization of the hidden state
		self.h0 = T.tanh(self.c0)
		
		# sequences: x_t
		# prior results: h_tm1, c_tm1
		# non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_co, b_y
		
		#step function for lstm 
		def one_lstm_step(x_t, h_tm1, c_tm1):#, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y):
		    #x_t = x_t.reshape((1,48,64,3))
		    x_t = x_t.dimshuffle('x',0,1,2)       
		    outputs = []     		    
		    for channel in range(len(channelLayers)):
		       inputStructure = channels[channel][0]
		       for layer in range(len(channelLayers[channel])):
		          if layer == 0:
		             channelLayers[channel][layer].input = InputLayer(x_t, inputStructure,  1).output                          
		          else:
		             channelLayers[channel][layer].input = channelLayers[channel][layer-1].output
                     
                              
		          channelLayers[channel][layer].getOutput()
		          #print "Channel:", channel, " - Layer:", layer
		          #print " ---- Input shape:",  channelLayers[channel][layer].image_shape          
		          #print " ---- Output shape:",  channelLayers[channel][layer].outputShape
		          output =  channelLayers[channel][layer].output.flatten(1)          
		       outputs.append(output)
                      
		    self.Conv2Output = theano.tensor.concatenate(outputs)
		    #self.Conv2Output = output
		    #print "Dimension output Shape:", self.Conv2Output.ndim
      
		    i_t = Activations.applyActivationFunction(self.activationFunction, theano.dot(self.Conv2Output, self.W_xi) + theano.dot(h_tm1, self.W_hi) + theano.dot(c_tm1, self.W_ci) + self.b_i)
		    f_t = Activations.applyActivationFunction(self.activationFunction,theano.dot(self.Conv2Output, self.W_xf) + theano.dot(h_tm1, self.W_hf) + theano.dot(c_tm1, self.W_cf) + self.b_f)
		    c_t = f_t * c_tm1 + i_t * self.act(theano.dot(self.Conv2Output, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c) 
		    o_t = Activations.applyActivationFunction(self.activationFunction,theano.dot(self.Conv2Output, self.W_xo)+ theano.dot(h_tm1, self.W_ho) + theano.dot(c_t, self.W_co)  + self.b_o)
		    h_t = o_t * self.act(c_t)		
		    y_t = self.sigma(theano.dot(h_t, self.W_hy) + self.b_y) 
		    self.h0 = h_t
		    return [h_t, c_t, y_t]#output of the LSTM block
				
		  
		# hidden and outputs of the entire sequence
		#[self.h_vals, _, self.y_vals], _ = theano.scan(fn=one_lstm_step, 
		                                  #sequences = dict(input=self.v, taps=[0]), 
		                                  #outputs_info = [self.h0, self.c0, None ], # corresponds to return type of fn
		                                  #non_sequences = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.W_hy, self.b_y] )
		[self.h_values, self.c0_vals, self.output], _ = theano.scan(fn=one_lstm_step, 
		                                  sequences = dict(input=input[0], taps=[0]),
		                                  outputs_info = [self.h0, self.c0, None] )# corresponds to return type of fn		                                  	
		#self.output = self.output[-1]
		self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.c0, self.W_hy, self.b_y]
		#self.params = [self.W_xi]
		self.output = T.argmax(self.output, axis=1)
		self.target = T.vector(dtype=self.dtype)
		
		self.cost = -T.mean(self.target * T.log(self.output)+ (1.- self.target) * T.log(1. - self.output))
		
        
	def sample_weights(self, sizeX, sizeY):
	    values = numpy.ndarray([sizeX, sizeY], dtype=self.dtype)
	    for dx in xrange(sizeX):
	        vals = numpy.random.uniform(low=-1., high=1.,  size=(sizeY,))
	        #vals_norm = np.sqrt((vals**2).sum())
	        #vals = vals / vals_norm
	        values[dx,:] = vals
	    _,svs,_ = numpy.linalg.svd(values)
	    #svs[0] is the largest singular value                      
	    values = values / svs[0]
	    return values    

	def errors(self, y):	    
	    return T.mean(T.neq(self.output, y))	             
     


class DeconvLayer(object):

    def __init__(
        self,
        flt_shape,
        img_shape,
        activationFunction,
        input=None,
        W=None,
        inhibitionFilters=None,
        W_prime=None,
        bhid=None,
        brec=None,
        poolsize=(2, 2)
    ):
        self.filter_shape = flt_shape
        self.image_shape = img_shape
        self.poolsize = poolsize
        self.activationFunction = activationFunction

        numpy_rng = numpy.random.RandomState(randint(0, 2 ** 30))

        if not W:
            fan_in = numpy.prod(flt_shape[1:])
            fan_out = (flt_shape[0] * numpy.prod(flt_shape[2:]))

            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                numpy.asarray(
                    numpy_rng.uniform(
                        low=-W_bound, high=W_bound, size=flt_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if inhibitionFilters is not None:
            rng = numpy.random.RandomState(89677)
            self.decayTerm = rng.uniform(low=0, high=1, size=[1])
            self.wInhibitory = inhibitionFilters[0].astype(theano.config.floatX)
            W = (W / (self.decayTerm + self.wInhibitory)).astype(theano.config.floatX)

        if not W_prime:
            W_prime = W[:, :, ::-1, ::-1].dimshuffle(1, 0, 2, 3)

        if not bhid:
            b_values = numpy.zeros((self.filter_shape[0],),
                                   dtype=theano.config.floatX)
            bhid = theano.shared(value=b_values, borrow=True)

        if not brec:
            b_values = numpy.zeros((self.filter_shape[1],),
                                   dtype=theano.config.floatX)
            brec = theano.shared(value=b_values, borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = brec
        # tied weights, therefore W_prime is W transpose
        # W_prime is W flipped because of the properties of CNN filters
        self.W_prime = W_prime

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        #conv_out = GpuCorrMM(border_mode='half')(input, self.W)
        x_shape = self.filter_shape[2]//2
        y_shape = self.filter_shape[3]//2
        conv_out = conv2d(
                    input=input,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    image_shape=self.image_shape,
                    border_mode='full'
                )[:,
                  :,
                  x_shape:-x_shape,
                  y_shape:-y_shape]

        return self.activationFunction(
                    conv_out
                    + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_reconstructed_input(self, input):
        #conv_out = GpuCorrMM(border_mode='half')(input, self.W_prime)
        x_shape = self.filter_shape[2]//2
        y_shape = self.filter_shape[3]//2
        conv_out = conv2d(
                    input=input,
                    filters=self.W_prime,
                    filter_shape=(self.filter_shape[1], self.filter_shape[0]) + self.filter_shape[2:],
                    image_shape=(self.image_shape[0], self.filter_shape[0]) + self.image_shape[2:],
                    border_mode='full'
                )[:,
                  :,
                  x_shape:-x_shape,
                  y_shape:-y_shape]
        return self.activationFunction(
                    conv_out
                    + self.b_prime.dimshuffle('x', 0, 'x', 'x'))

    def max_pool(self, input):
        self.pool_in = input
        self.pool_out = downsample.max_pool_2d(
            input=input,
            ds=self.poolsize,
            ignore_border=True
        )
        return self.pool_out

    def get_forward_pass(self, input):
        y = self.get_hidden_values(input)
        z = self.max_pool(y)
        return z

    def unpool(self, input):
        unpool = T.grad(T.sum(self.pool_out), wrt=self.pool_in) * \
                        repeat(repeat(input,
                                      self.poolsize[0],
                                      2),
                               self.poolsize[1],
                               3)
        return unpool

    def get_backward_pass(self, input):
        y = self.unpool(input)
        z = self.get_reconstructed_input(y)

        return z