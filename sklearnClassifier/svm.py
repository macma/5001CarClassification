import svm_feature_loader as sfl
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np



data = sfl.svm_feature_loader()
features = np.asarray(data['featureList'])
target = np.asarray(data['labelList'])
pca = PCA(n_components=7)
feature_pca = pca.fit_transform(features)
features = feature_pca
print sum(pca.explained_variance_ratio_)
'''

'''
#Normalization
#Subtract the mean for each feature
features -= np.mean(features, axis=0)
#Divide each feature by its standard deviation
features /= np.std(features, axis=0)


#n-fold cross validation

#5 Fold Cross Validation
kf = KFold(n=len(target), n_folds=5, shuffle=True)

cv = 0
acc = []
models = []
for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]

    #Training Logistic Regression
    # model = LogisticRegression()
    # model.fit(tr_features, tr_target)

    #Training SVM Model
    model = SVC(probability=True)
    model.fit(tr_features, tr_target)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(np.argmax(model.predict_proba(tr_features),axis=1) == tr_target)
    tst_accuracy = np.mean(np.argmax(model.predict_proba(tst_features),axis=1) == tst_target)

    print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    acc.append(tst_accuracy)
    models.append(model)
    cv += 1



'''
top 3 eigenvector explains > 90% of the total variance, PCA is validated.
'''