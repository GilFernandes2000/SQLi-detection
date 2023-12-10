import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn import ensemble
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import sys
import warnings
warnings.filterwarnings('ignore')

def waitforEnter():
    input("Press ENTER to continue.")
            
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    plt.close()

def distance(c,p):
    s=0
    n=0
    for i in range(len(c)):
        if c[i]>0:
            s+=np.square((p[i]-c[i])/c[i])
            n+=1
    
    return(np.sqrt(s/n))



Classes = {0:'Normal', 1:'SQLInjection'}
plt.ion()
nfgi = 1

            # Get the features and classes from the files #
features_Normal1 = np.loadtxt('features/normal2_features.dat')
features_Normal2 = np.loadtxt('features/normal1_features.dat')
features_Normal3 = np.loadtxt('features/normal3_features.dat')
features_SQLi = np.loadtxt('features/blindboolean_features.dat')

oClass_Normal = np.ones((len(features_Normal2)+len(features_Normal3),1))*0
oClass_NormalTest = np.ones((len(features_Normal1),1))*0
oClass_SQLi = np.ones((len(features_SQLi),1))*1

features = np.vstack((features_Normal2,features_Normal3,features_SQLi))
oClass = np.vstack((oClass_Normal,oClass_SQLi))

print("train size: {}".format(features.shape))
print("Class size: {}".format(oClass.shape))

                    # Train the model #

trainFeaturesNormal = np.vstack((features_Normal2,features_Normal3))
trainClass = oClass_Normal

testFeaturesSQLi = features_SQLi
testClassSQLi = oClass_SQLi

testFeaturesNormal = features_Normal1
testClassNormal = oClass_NormalTest

scaler = MaxAbsScaler().fit(trainFeaturesNormal)

trainFeaturesNormal=scaler.transform(trainFeaturesNormal)
testFeaturesSQLi=scaler.transform(testFeaturesSQLi)
testFeaturesNormal=scaler.transform(testFeaturesNormal) 

                            # PCA #

""" pca = PCA(n_components=28, svd_solver='full')

trainPCA = pca.fit_transform(trainFeaturesNormal)
trainFeaturesNormal = pca.transform(trainFeaturesNormal)

testFeaturesSQLi = pca.fit_transform(testFeaturesSQLi)
testFeaturesNormal = pca.transform(testFeaturesNormal)
 """

                            # One Class Support Vector #
print('\n-- Anomaly Detection based on One Class Support Vector Machines--')

testFeatures = np.vstack((testFeaturesSQLi,testFeaturesNormal))
testClass = np.vstack((testClassSQLi,testClassNormal))  

nu=0.1
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(trainFeaturesNormal, trainClass)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(trainFeaturesNormal, trainClass)  
poly_ocsvm = svm.OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=4).fit(trainFeaturesNormal, trainClass)  

L1=ocsvm.predict(testFeatures)
L2=rbf_ocsvm.predict(testFeatures)
L3=poly_ocsvm.predict(testFeatures)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=testFeatures.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


