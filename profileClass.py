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
features_Normal = np.loadtxt('normal1_features.dat')
features_SQLi = np.loadtxt('SQLiBasic_features.dat')

oClass_Normal = np.ones((len(features_Normal),1))*0
oClass_SQLi = np.ones((len(features_SQLi),1))*1

features = np.vstack((features_Normal,features_SQLi))
oClass = np.vstack((oClass_Normal,oClass_SQLi))

plt.figure(2)
plotFeatures(features,oClass,4,10)
plt.figure(3)
plotFeatures(features,oClass,0,7)

percentage=0.5
pN=int(len(features_Normal)*percentage)
trainFeatures_normal=features_Normal[:pN,:]
pSQL=int(len(features_SQLi)*percentage)
trainFeatures_sqli=features_SQLi[:pN,:]

i2train = trainFeatures_normal
o2trainClass = oClass_Normal[:pN,:]

i3Ctrain = np.vstack((trainFeatures_normal,trainFeatures_sqli))
o3CtrainClass = np.vstack((oClass_Normal[:pN,:],oClass_SQLi[:pSQL,:]))

testFeatures_normal=features_Normal[pN:,:]
testFeatures_sqli=features_SQLi[pSQL:,:]

iAtest = np.vstack((testFeatures_normal,testFeatures_sqli))
o3testClass = np.vstack((oClass_Normal[pN:,:],oClass_SQLi[pSQL:,:]))


print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i2train=np.vstack((trainFeatures_normal))
i3Atest=np.vstack((testFeatures_normal,testFeatures_sqli))
#scaler = MaxAbsScaler().fit(i2train)
#i2train=scaler.transform(i2train)

#scaler = MaxAbsScaler().fit(i3Atest)
#i3Atest=scaler.transform(i3Atest)

nu=0.1
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i2train)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i2train)  
poly_ocsvm = svm.OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=4).fit(i2train)  

L1=ocsvm.predict(i3Atest)
L2=rbf_ocsvm.predict(i3Atest)
L3=poly_ocsvm.predict(i3Atest)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3Atest.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
