import pandas as pd
from sklearn.model_selection import train_test_split
from math import *
from sklearn.decomposition import PCA



data = pd.read_csv("galex.csv")

feature_cols = ['ra','dec','u','g','r','i','z','nuv_mag','u-g','u-r','u-i','u-z','u-nuv_mag','g-r','g-i','g-z','g-nuv_mag','r-i','r-z','r-nuv_mag','i-z','i-nuv_mag','z-nuv_mag']
target_var=['class']
X = data[feature_cols] # Features
y = data[target_var] # Target variable
y_pred=[]
pca = PCA(n_components=8).fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(pca, y, test_size=0.3,random_state=180 ) # 70% training and 30% test
k=23 #k>number of classes , k is not a multiple of number of classes


def predict():
    
    for i in range(len(x_test)):
        result=[]
        pred=[]
        for j in range(len(x_train)):
            euclidean_distance=0
            for m in range(8):
                euclidean_distance=euclidean_distance+pow((x_test[i][m]-x_train[j][m]),2)
            euclidean_distance=sqrt(euclidean_distance)
            result.append([euclidean_distance,j])
        dup=[]
        dup=result
        dup.sort(key= lambda x: x[0])
        for l in range(k):
            #ind=result.index(dup[l])
            pred.append(y_train.iloc[dup[l][1],0])
        y_pred.append(max(set(pred),key=pred.count))

def accuracy():
    count=0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test.iloc[i,0]):
            count+=1
    print("Accuracy :",(count/len(y_test))*100)

predict()
accuracy()
