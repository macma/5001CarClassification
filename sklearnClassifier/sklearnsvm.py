#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:33:41 2016

@author: mac
"""

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
import svm_feature_loader as sfl
#from sklearn.decomposition import PCA

data = sfl.svm_feature_loader()
features = np.asarray(data['featureList'])
target = np.asarray(data['labelList'])

## We load the data with load_iris from sklearn
datai = load_iris()
featuresi = datai['data']
feature_namesi = datai['feature_names']
targeti = datai['target']
print type(features), type(featuresi)
#Normalization
#Subtract the mean for each feature
features -= np.mean(features, axis=0)
#Divide each feature by its standard deviation
features /= np.std(features, axis=0)


#single logisticRegression
##Binary label
#is_versicolor = target == 1
#binary_target = np.zeros(len(target))
#binary_target[is_versicolor] = 1
#
##Training Logistic Regression
#lr = LogisticRegression()
#lr.fit(features, binary_target)
#
##Measuring accuracy
#accuracy = np.mean(lr.predict(features) == binary_target)
#print "Training Accuracy: %f" % accuracy




#n-fold cross validation
#Binary label
is_versicolor = target == 1
binary_target = np.zeros(len(target))
binary_target[is_versicolor] = 1

#5 Fold Cross Validation
kf = KFold(n=len(binary_target), n_folds=5, shuffle=True)

cv = 0
for tr, tst in kf:

    #Train Test Split
    tr_features = features[tr, :]
    tr_target = binary_target[tr]

    tst_features = features[tst, :]
    tst_target = binary_target[tst]

    #Training Logistic Regression
    # model = LogisticRegression()
    # model.fit(tr_features, tr_target)

    #Training SVM Model
    model = SVC()
    model.fit(tr_features, tr_target)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)

    print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    cv += 1






