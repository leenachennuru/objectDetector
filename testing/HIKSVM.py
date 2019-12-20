"""
@author: 4chennur, 4wahab
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.externals import joblib
import utils
import math
from sklearn import metrics

svmTrainDataPath ="SVM/TrainingDataSiftSpatialPyramid10000.npy"
svmTrainLabelPath = "SVM/TrainingLabelsSiftSpatialPyramid10000.npy"

# histogram intersection kernel
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

def SVM_HIK(trainDataPath, trainLabelPath, kernelType):
    trainData = np.array(np.load(trainDataPath))
    trainLabels = np.load(trainLabelPath)


    if kernelType == "HIK":

        #gramMatrix = histogramIntersection(trainData, trainData)
        clf = SVC(kernel=histogramIntersection)
        clf.fit(trainData, trainLabels)
	joblib.dump(clf, 'SVM/SVMHIKSPCustom10000'+'.pkl')
        #predictMatrix = histogramIntersection(testData, trainData)
        SVMResults = clf.predict(trainData)
	print 'Precision Score: ' + str(metrics.precision_score(trainLabels, SVMResults))
	print 'Recall Score: ' + str(metrics.recall_score(trainLabels, SVMResults))
        correct = sum(1.0 * (SVMResults == trainLabels))
        accuracy = correct / len(trainLabels)
        print "SVM (Histogram Intersection): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(trainLabels))+ ")"


#SVM_HIK(svmTrainDataPath,svmTrainLabelPath,"HIK")

#svm = joblib.load("SVM/SVMHIKSP10000.pkl")
#print "SVM is loaded"
#trainData = np.array(np.load(svmTrainDataPath))
#trainLabels = np.load(svmTrainLabelPath)
#print "Training Data is loaded"
#gramMatrix = histogramIntersection(trainData, trainData)
#print "gramMatrix is computed"
#SVMResults = svm.predict(gramMatrix)
#print 'Precision Score: ' + str(metrics.precision_score(trainLabels, SVMResults))
#print 'Recall Score: ' + str(metrics.recall_score(trainLabels, SVMResults))
