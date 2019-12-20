# -*- coding: utf-8 -*-

from Utils import DataUtil
import MCCNNExperiments

def loadExperiment():


    modelDirectory = ""

    networkTopology, trainingParameters, experimentParameters, visualizationParameters,networkState = DataUtil.loadNetworkState(modelDirectory)

    print networkState

    saveNetworkParameters = [False]

   # MCCNNExperiments.runExperiment(networkTopology,trainingParameters, experimentParameters, visualizationParameters, saveNetworkParameters )




loadExperiment()
