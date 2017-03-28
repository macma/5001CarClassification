import svm_feature_loader as sfl
from sklearn.decomposition import PCA
import numpy as np
data = sfl.svm_feature_loader()
features = data['featureList']
labels = data['labelList']
pca = PCA()
pca.fit(features)
print pca.explained_variance_ratio_
'''
top 3 eigenvector explains > 90% of the total variance, PCA is validated.
'''