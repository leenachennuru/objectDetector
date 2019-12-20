"""
@author: 4chennur, 4wahab
"""
import cv2
from sklearn.externals import joblib
import sklearn.cluster as cluster
from sklearn import svm
import os
import numpy as np
import glob
import detectorDescriptor2 as detDes

def KMeans(codeBook, noClusters):
	k_means =cluster.KMeans(init='k-means++', n_clusters=noClusters, n_init=10, n_jobs=10,verbose = True)
	k_means.fit(codeBook)
	joblib.dump(k_means, 'CrossValidationData/kmeans'+ str(noClusters) +'.pkl')
	return k_means

def miniBatchCluster(codeBook, noClusters):
	mbk = cluster.MiniBatchKMeans(init='k-means++', n_clusters=noClusters,
                batch_size=noClusters * 3, n_init=3, max_iter=50,
                max_no_improvement=3, verbose=True, compute_labels=False)
	mbk.fit(codeBook)
	joblib.dump(mbk, 'CrossValidationData/kmeans'+ str(noClusters) +'.pkl')
	return mbk

def minDistance(kMeansModel, featureList):
	return kMeansModel.predict(featureList)

def svmTraining(TrainingData, TrainingLabels, cValue, sValue):
	estimator = svm.SVC(C= cValue, gamma = sValue, verbose=True)
	estimator.fit(TrainingData, TrainingLabels)
	joblib.dump(estimator, 'CrossValidationData/svm'+str(cValue) + str(sValue) +'.pkl')
	return estimator

def creatingTrainingSVM(codeBookEstimator, codeBookCenters):
	rootInputName = "FinalExtensiveData/"
	formatName = "*.png"

	listDir = [];
	listDir.append("kettle-bb/")
	listDir.append("milk-bb/")
	listDir.append("basket-bb/")
	listDir.append("mug-bb/")
	listDir.append("kettle-bgbb/")
	listDir.append("milk-bgbb/")
	listDir.append("basket-bgbb/")
	listDir.append("mug-bgbb/")
	listDir.append("background/")

	histogramSVM = []
	labels = []
	HistogramComputed = 0
	# testing the things
	dirCount = 0
	for direct in listDir:
		fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
		print ("Current Directory Name:" + rootInputName + listDir[dirCount])
		count = 0
		for files in fileList:
			inputImage=cv2.imread(fileList[count])
			fileName = os.path.basename(fileList[count])
			roiImageFiltered = inputImage
			kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
			kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
			if np.size(kp)>0:
				clusterEval = codeBookEstimator.predict(des)
				histPoints = np.histogram(clusterEval, bins=np.arange(codeBookCenters), density=False)
				histogramSVM.append(histPoints[0])
				labels.append([dirCount])
				HistogramComputed = HistogramComputed + 1
				print ("Histogram computed for the chosen Image:" + str(count) + ' Category:' + str(dirCount))
			count = count + 1
		dirCount = dirCount + 1
	histogramNew  = np.float32(np.array(histogramSVM))
	labelsNew  = np.float32(np.array(labels))

	np.save("CrossValidationData/TrainingData"+str(codeBookCenters) +".npy", histogramNew)
	np.save("CrossValidationData/TrainingLabels"+str(codeBookCenters) +".npy", labelsNew)
	return histogramNew, labelsNew
