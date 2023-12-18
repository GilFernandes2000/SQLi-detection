import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn import ensemble
from sklearn import metrics
import time
import sys
import warnings
warnings.filterwarnings('ignore')

def waitforEnter():
    input("Press ENTER to continue.")
            
def plotFeatures(features,oClass,f1index=0,f2index=1,labelX='',labely='',title=''):
    nObs,nFea=features.shape
    colors=['g','r','b']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.title(title)
    plt.xlabel(labelX)
    plt.ylabel(labely)
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

def confusionMatrix(actual, predicted, title):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i] and actual[i] == -1:
            tp += 1
        elif actual[i] == predicted[i] and actual[i] == 1:
            tn += 1
        elif actual[i] != predicted[i] and actual[i] == -1:
            fp += 1
        else:
            fn += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall)) if precision+recall != 0 else 0
    print("Accuracy: ",accuracy*100,"%")
    print("Precision: ",precision*100,"%")
    print("Recall: ",recall*100,"%")
    print("F1-score: ",f1)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, False])

    cm_display.plot()
    plt.title(title)
    plt.show()
    waitforEnter()
    plt.close()

Classes = {0:'Normal', 1:'SQLInjection'}
plt.ion()

            # Get the features and classes from the files #
features_Normal = np.loadtxt('features/normal_features.dat')
features_Normal1 = np.loadtxt('features/normal1_features.dat')
features_Normal2 = np.loadtxt('features/normal2_features.dat')
features_Normal3 = np.loadtxt('features/normal3_features.dat')
features_SQLi = np.loadtxt('features/SQLItimebased_features.dat')

oClass_NormalTest = np.ones((len(features_Normal2)+len(features_Normal3),1))*0
oClass_Normal = np.ones((len(features_Normal),1))*0
oClass_SQLi = np.ones((len(features_SQLi),1))*1

features = np.vstack((features_Normal,features_SQLi))
oClass = np.vstack((oClass_Normal,oClass_SQLi))

print("train size: {}".format(features.shape))
print("Class size: {}".format(oClass.shape))


plt.figure(2)
plotFeatures(features,oClass,1,2, 'Mean', 'Standard Deviation', 'Duration')
plt.figure(3)
plotFeatures(features,oClass,3,4, 'Min', 'Max', 'Duration')
plt.figure(4)
plotFeatures(features,oClass,5,6, 'Mean', 'Standard Deviation', 'Packets')
plt.figure(5)
plotFeatures(features,oClass,7,8, 'Min', 'Max', 'Packet')
plt.figure(6)
plotFeatures(features,oClass,9,10, 'Mean', 'Standard Deviation', 'Down Bytes')
plt.figure(7)
plotFeatures(features,oClass,11,12, 'Min', 'Max', 'Down Bytes')
plt.figure(8)
plotFeatures(features,oClass,13,14, 'Mean', 'Standard Deviation', 'Up Bytes')
plt.figure(9)
plotFeatures(features,oClass,15,16, 'Min', 'Max', 'Up Bytes')
plt.figure(10)
plotFeatures(features,oClass,17,18, 'Mean', 'Standard Deviation', 'Packet Interarrival Time')
plt.figure(11)
plotFeatures(features,oClass,19,20, 'Min', 'Max', 'Packet Interarrival Time') 
plt.figure(12)
plotFeatures(features,oClass,21,22, 'Mean', 'Standard Deviation', 'Time Between Flows')
plt.figure(13)
plotFeatures(features,oClass,23,24, 'Min', 'Max', 'Time Between Flows')
plt.figure(14)
plotFeatures(features,oClass,25,26, 'Mean', 'Standard Deviation', 'Ratio')
plt.figure(15)
plotFeatures(features,oClass,27,28, 'Min', 'Max', 'Ratio') 


                    # Train the model #

trainFeaturesNormal = np.vstack((features_Normal))
trainClass = oClass_Normal

testFeaturesSQLi = features_SQLi
testClassSQLi = oClass_SQLi

testFeaturesNormal = np.vstack((features_Normal2,features_Normal3))
testClassNormal = oClass_NormalTest 

scaler = MaxAbsScaler().fit(trainFeaturesNormal)
scaler = StandardScaler().fit(trainFeaturesNormal)

trainFeaturesNormal=scaler.transform(trainFeaturesNormal)
testFeaturesSQLi=scaler.transform(testFeaturesSQLi)
testFeaturesNormal=scaler.transform(testFeaturesNormal)  
 
                            # PCA #

pca = PCA(n_components=29, svd_solver='full')

trainPCA = pca.fit_transform(trainFeaturesNormal)
trainFeaturesNormal = pca.transform(trainFeaturesNormal)

testFeaturesSQLi = pca.fit_transform(testFeaturesSQLi)
testFeaturesNormal = pca.transform(testFeaturesNormal) 


testFeatures = np.vstack((testFeaturesSQLi,testFeaturesNormal))
testClass = np.vstack((testClassSQLi,testClassNormal))  

plt.figure(1)
plotFeatures(testFeatures,testClass,17,18, 'Mean', 'Standard Deviation', 'Duration')


                           # Centroid Distances #

centroids={}
for c in range(2):  # Only the first two classes
    pClass=(trainClass==c).flatten()
    centroids.update({c:np.mean(trainFeaturesNormal[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

AnomalyThreshold = 1
pred = []
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=testFeatures.shape
for i in range(nObsTest):
    x=testFeatures[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        pred.append(-1)
    else:
        pred.append(1)
        result="OK"
       
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[testClass[i][0]],*dists,result))

actual = np.concatenate((np.ones(len(testClassSQLi))*-1, np.ones(len(testClassNormal))))

print('\n-- Centroid Statistics --')
confusionMatrix(actual, pred, "Centroid Heatmap") 




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

actual = np.concatenate((np.ones(len(testClassSQLi))*-1, np.ones(len(testClassNormal))))

print('\n-- Kernel Linear Statistics --')
confusionMatrix(actual,L1, "Kernel Linear Heatmap")

print('\n-- Kernel RBF Statistics--')
confusionMatrix(actual,L2, "Kernel RBF Heatmap")

print('\n-- Kernel Poly Statistics--')
confusionMatrix(actual,L3, "Kernel Poly Heatmap") 



                        # Isolation Forest #

print('\n-- Anomaly Detection based on Isolation Forest --')
tree = IsolationForest(contamination=0.3, random_state=50)
tree.fit(trainFeaturesNormal)
pred = tree.predict(testFeatures)
AnomResults={-1:"Anomaly",1:"OK"}
nObsTest,nFea=testFeatures.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): {}'.format(i,Classes[testClass[i][0]],AnomResults[pred[i]]))

print(pred)

print('\n-- Isolation Forest Statistics --')

actual = np.concatenate((np.ones(len(testClassSQLi))*-1, np.ones(len(testClassNormal))))

confusionMatrix(actual,pred, "Isolation Forest Heatmap")


                        # Local Outlier Factor #

print('\n-- Anomaly Detection based on Local Outlier Factor --')

lof = LocalOutlierFactor(n_neighbors=15, contamination=0.3, novelty=True)
lof.fit(trainFeaturesNormal)
pred = lof.predict(testFeatures)
AnomResults={-1:"Anomaly",1:"OK"}
nObsTest,nFea=testFeatures.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): {}'.format(i,Classes[testClass[i][0]],AnomResults[pred[i]]))

print('\n-- LOF Statistics --')

actual = np.concatenate((np.ones(len(testClassSQLi))*-1, np.ones(len(testClassNormal))))

confusionMatrix(actual,pred, "LOF Heatmap") 

