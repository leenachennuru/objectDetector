"""
@author: 4chennur, 4wahab
"""
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib

TrainingData = np.load("SVM/TrainingDataSubsampled75000.npy")
TrainingLabels = np.load("SVM/TrainingLabelsSubsampled75000.npy")
clf = svm.SVC(verbose =10)
clf.fit(TrainingData,TrainingLabels)
y_pred = clf.predict(TrainingData)
joblib.dump(clf, 'SVM/SVMTrainedDefaultSubsampled75.pkl')

print 'Precision: ' + str(metrics.precision_score(TrainingLabels, y_pred)) + '%'
print 'Recall: ' + str(metrics.recall_score(TrainingLabels, y_pred)) + '%'
