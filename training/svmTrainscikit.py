"""
@author: 4chennur, 4wahab
"""
import cv2
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics
import numpy as np
import sklearn.cross_validation as cross
import sklearn.grid_search as gridSearch
import clusteringSk as cl
from sklearn.externals import joblib


#X = np.load('SVM/TrainingData.npy')
#Y = np.load('SVM/TrainingLabels.npy')
#Y_new = utils.column_or_1d(Y, warn=True)
#clf = svm.SVC(verbose = True)
#clf.fit(X, Y_new)
#
#y_pred = clf.predict(X)
#print metrics.accuracy_score(Y_new, y_pred)

#X = np.load('CodeBook/RandomCodeBook.npy')[0:10000, :]
#codebooksize = 100
#
#mbk = cluster.MiniBatchKMeans(init='k-means++', n_clusters=codebooksize,
#                batch_size=codebooksize * 3, n_init=3, max_iter=50,
#                max_no_improvement=3, verbose=1, compute_labels=False)
#b = mbk.fit(X)
#labels  = mbk.predict(X)
#joblib.dump(mbk, 'kmeans.pkl')
#print labels[0:10]
#
#mbk = joblib.load('kmeans.pkl')
#
#abc = cl.minDistance(mbk, X[0:10,:])



###################################################################################################
#iris = load_iris()
#X = iris.data
#Y = iris.target
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = cross.StratifiedShuffleSplit(Y, n_iter=5, test_size=0.2, random_state=42)
#grid = gridSearch.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose = 1, n_jobs = 4)
#grid.fit(X, Y)
#print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))


##################################################################################################

# CodeBook 100000 #
codeBook = np.load('CodeBook/withNoise.npy')
C_range = np.logspace(-2, 6, 20)
gamma_range = np.logspace(-3, 3, 20)
param_grid = dict(gamma=gamma_range, C=C_range)


codeBookCenters = 100000
codeBookEstimator = cl.KMeans(codeBook, codeBookCenters)
TrainingData, TrainingLabels = cl.creatingTrainingSVM(codeBookEstimator, codeBookCenters)
cv = cross.StratifiedShuffleSplit(TrainingLabels, n_iter=5, test_size=0.2, train_size=0.8, random_state=42)
grid = gridSearch.GridSearchCV(svm.SVC(verbose=10, class_weight = 'auto'), param_grid=param_grid, cv=cv, verbose = 1, n_jobs = 6)
joblib.dump(grid, 'CrossValidationData/gridCross'+ str(codeBookCenters) +'.pkl')

del codeBookCenters, codeBookEstimator,TrainingData, TrainingLabels, cv, grid

codeBookCenters = 75000
codeBookEstimator = cl.KMeans(codeBook, codeBookCenters)
TrainingData, TrainingLabels = cl.creatingTrainingSVM(codeBookEstimator, codeBookCenters)
cv = cross.StratifiedShuffleSplit(TrainingLabels, n_iter=5, test_size=0.2, train_size=0.8, random_state=42)
grid = gridSearch.GridSearchCV(svm.SVC(verbose=10, class_weight = 'auto'), param_grid=param_grid, cv=cv, verbose = 1, n_jobs = 6)
joblib.dump(grid, 'CrossValidationData/gridCross'+ str(codeBookCenters) +'.pkl')

del codeBookCenters, codeBookEstimator,TrainingData, TrainingLabels, cv, grid

codeBookCenters = 50000
codeBookEstimator = cl.KMeans(codeBook, codeBookCenters)
TrainingData, TrainingLabels = cl.creatingTrainingSVM(codeBookEstimator, codeBookCenters)
cv = cross.StratifiedShuffleSplit(TrainingLabels, n_iter=5, test_size=0.2, train_size=0.8, random_state=42)
grid = gridSearch.GridSearchCV(svm.SVC(verbose=10, class_weight = 'auto'), param_grid=param_grid, cv=cv, verbose = 1, n_jobs = 6)
joblib.dump(grid, 'CrossValidationData/gridCross'+ str(codeBookCenters) +'.pkl')

del codeBookCenters, codeBookEstimator,TrainingData, TrainingLabels, cv, grid

codeBookCenters = 25000
codeBookEstimator = cl.KMeans(codeBook, codeBookCenters)
TrainingData, TrainingLabels = cl.creatingTrainingSVM(codeBookEstimator, codeBookCenters)
cv = cross.StratifiedShuffleSplit(TrainingLabels, n_iter=5, test_size=0.2, train_size=0.8, random_state=42)
grid = gridSearch.GridSearchCV(svm.SVC(verbose=10, class_weight = 'auto'), param_grid=param_grid, cv=cv, verbose = 1, n_jobs = 6)
joblib.dump(grid, 'CrossValidationData/gridCross'+ str(codeBookCenters) +'.pkl')
