import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

data=pd.read_csv("voice.csv")

data=pd.DataFrame({"sfm":data.sfm,"sd":data.sd})

plt.scatter(data.sfm,data.sd)
plt.xlabel("sfm")
plt.ylabel("sd")
plt.title("Raw Data")
plt.show()

''' find best k value
wcss=[]
for k in range(1,15):
    k_means=KMeans(n_clusters=k)
    k_means.fit(data)
    wcss.append(k_means.inertia_)
plt.plot(range(1,15),wcss)
plt.show()
'''

k_means=KMeans(n_clusters=3)
data["label"]=k_means.fit_predict(data)

plt.scatter(data.sfm[data.label==0],data.sd[data.label==0],color="red",alpha=0.7)
plt.scatter(data.sfm[data.label==1],data.sd[data.label==1],color="green",alpha=0.45)
plt.scatter(data.sfm[data.label==2],data.sd[data.label==2],color="blue",alpha=0.2)
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],color="yellow",alpha=1)
plt.title("Finished Data")
plt.show()