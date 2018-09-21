import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("data.csv")
data["class"]=[0 if i=="Iris-setosa" else \
    (1 if i=="Iris-versicolor" else 2) for i in data["class"]]

x=data.drop(["class"],axis=1)
y=data["class"]

x=(x-np.min(x))/(np.max(x)-np.min(x))

x,x_test,y,y_test=train_test_split(x,y,test_size=0.1)

knn=KNeighborsClassifier(n_neighbors=3)

accuracies=cross_val_score(estimator=knn,X=x,y=y,cv=10)
print("Accuracy=",np.mean(accuracies))

knn.fit(x,y)
print("Test Accuracy=",knn.score(x,y))