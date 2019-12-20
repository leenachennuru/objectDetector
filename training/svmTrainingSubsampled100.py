"""
@author: 4chennur, 4wahab
"""
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib

codeBookCenters = np.load('CodeBook/codeBookNoiseCentersSubsampled100000.npy')

TrainingData = np.load("SVM/TrainingDataSubsampled100000.npy")
TrainingLabels = np.load("SVM/TrainingLabelsSubsampled100000.npy")
clf = svm.SVC(verbose =10)
clf.fit(TrainingData,TrainingLabels)
y_pred = clf.predict(TrainingData)
joblib.dump(clf, 'SVM/SVMTrainedDefaultSubsampled100.pkl')

print 'Precision: ' + str(metrics.precision_score(TrainingLabels, y_pred)) + '%'
print 'Recall: ' + str(metrics.recall_score(TrainingLabels, y_pred)) + '%'
