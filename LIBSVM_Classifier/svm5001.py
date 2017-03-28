#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:12:39 2016

@author: mac
"""
from svmutil import *
#print bc.convertbool('yes')
import datetime as dt
import svm_feature_loader as sfl
#from sklearn.decomposition import PCA
import numpy as np
data = sfl.svm_feature_loader()
features = data['featureList']
#features -= np.mean(features, axis=0)
#features /= np.std(features, axis=0)
#print features
labels = data['labelList']
#print len(features)

#
m = svm_train(labels, features, '-t 1 -c 1000 -v 5')
print 'done'
