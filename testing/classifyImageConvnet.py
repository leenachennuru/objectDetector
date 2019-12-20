# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""
from Utils import DataUtil
from Networks import MCCNN
import MCCNNExperiments
import cv2
import os
import glob


modelDirectoryCPU = "/informatik2/students/home/4wahab/Desktop/convNetDataset/experiments/convNetContoursResized20Epoch/model/repetition_0_BestTest_convNetContoursResized20Epoch_.save"

networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectoryCPU)



experimentParameters[0] = os.path.dirname(os.path.abspath(modelDirectoryCPU))
experimentParameters.append(False)

saveNetworkParameters = [False]
network = MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )

fileList = glob.glob('/informatik2/students/home/4wahab/Desktop/convNetDataset/convNetFullNormalizedDemo/testing/kettle-bb/'+'*.png')
result = []
for files in fileList:
	img = cv2.imread(files)
	""" image, DATA_MODALITY["Image"], imageSize"""
	img,frame = DataUtil.prepareDataLive(img, DataUtil.DATA_MODALITY["Image"], networkTopology[6][0][0][4])
	"""The model, image, batchSize"""
	result1 = MCCNN.classify(network[len(network)-1],[img],trainingParameters[4])[0]
	result.append(result1)
