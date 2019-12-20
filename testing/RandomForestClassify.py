# -*- coding: utf-8 -*-
"""
@author: 4chennur, 4wahab
"""

import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#from scipy import misc as msc
#import classifier
#import glob
#import preObj as prePro
#import detectorDescriptor as detDes
#import os

from sklearn.externals import joblib
import sklearn.cluster as cluster
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

svmTrainDataPath ="SVM/TrainingDataSiftwithHomeLab10000.npy"
svmTrainLabelPath = "SVM/TrainingLabelsSiftwithHomeLab10000.npy"

svmTrainData = np.load(svmTrainDataPath)
print 'Training data has loaded'
svmTrainLabels = np.load(svmTrainLabelPath)


#######################################   SVM Opencv Implementation     #############################################################

#svm = cv2.SVM()
#svm_params = dict( kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, C=1, gamma=5.383, class_weights = [0.1375, 0.1375, 0.1375, 0.1375, 0.1, 0.1, 0.1, 0.1, 0.05] )
#svm.train(svmTrainData, svmTrainLabels, params = svm_params)
#results = svm.predict_all(svmTrainData)
#computeAccuracy = results==svmTrainLabels
#print ("Accuracy of the SVM on Training Data is: ", np.float32(np.count_nonzero(computeAccuracy))/np.float32(np.size(computeAccuracy,0))*100)
#svm.save("SVM/svmNoise25000finalC1weightsadded.dat")

####################################### SVM Scikit learn implementation   ############################################################

def svmTraining(TrainingData, TrainingLabels):
    print 'SVM has started training'
    estimator = svm.SVC(verbose=True,class_weight="auto")
    estimator.fit(TrainingData, TrainingLabels)
    y_pred = estimator.predict(TrainingData)
    print 'Precision Score: ' + str(metrics.precision_score(TrainingLabels, y_pred))
    print 'Recall Score: ' + str(metrics.recall_score(TrainingLabels, y_pred))
    joblib.dump(estimator, 'SVM_sift75000'+'.pkl')
    return estimator

def randomForest(TrainingData, TrainingLabels):
    print 'Random Forest classifier has started to train'
    randFor = RandomForestClassifier(n_estimators=20, oob_score = True, class_weight = 'balanced' )
    randFor = randFor.fit(TrainingData, TrainingLabels)
    y_pred = randFor.predict(TrainingData)
    print 'Precision Score: ' + str(metrics.precision_score(TrainingLabels, y_pred))
    print 'Recall Score: ' + str(metrics.recall_score(TrainingLabels, y_pred))
    print "Out of bag error is" + str(randFor.oob_score_)
    #print randFor.feature_importances_
    joblib.dump(randFor,'SVM/RandomForestsiftwithHomeLab10000.pkl')
    return randFor

randFor = randomForest(svmTrainData,svmTrainLabels)
