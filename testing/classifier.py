"""
@author: 4chennur, 4wahab
"""

import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import misc as msc
import scipy.spatial.distance as computeMod

def kmeansCodeBook(features,K):
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 10)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_PP_CENTERS
    attempts = 10
    compactness, labels, centers = cv2.kmeans(features, K, criteria, attempts, flags)
    return compactness, labels, centers


def minDistance(descriptorSet, clusterCenters):
	distMatrix = computeMod.cdist(descriptorSet,clusterCenters, 'euclidean')
	minPoints= np.argmin(distMatrix, axis =1)
	histPoints = np.histogram(minPoints, bins=np.arange(np.size(clusterCenters,0)+1), density=False)
	return histPoints


class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_RBF,
                       svm_type = cv2.SVM_C_SVC,  C=2.67, gamma=5.383,
                        )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])


'''Uncomment when giving it training data. Remaining is self explanatory
clf = SVM()
clf.train(samples, y_train)
y_val = clf.predict(samples)
'''
