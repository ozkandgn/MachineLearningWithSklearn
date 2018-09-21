import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

data=pd.read_csv("voice.csv")

data=pd.DataFrame({"sfm":data.sfm,"sd":data.sd})

plt.scatter(data.sfm,data.sd)
plt.xlabel("sfm")
plt.ylabel("sd")
plt.title("Raw Data")
plt.show()


hiyerartical_c=AgglomerativeClustering(n_clusters=5,affinity= "euclidean",linkage= "ward")
data["label"]=hiyerartical_c.fit_predict(data)

plt.scatter(data.sfm[data.label==0],data.sd[data.label==0],color="red",alpha=0.6)
plt.scatter(data.sfm[data.label==1],data.sd[data.label==1],color="green",alpha=0.45)
plt.scatter(data.sfm[data.label==2],data.sd[data.label==2],color="blue",alpha=0.3)
plt.scatter(data.sfm[data.label==3],data.sd[data.label==3],color="brown",alpha=0.2)
plt.scatter(data.sfm[data.label==4],data.sd[data.label==4],color="gray",alpha=0.2)
plt.scatter(hiyerartical_c.cluster_centers_[:,0],hiyerartical_c.cluster_centers_[:,1],color="yellow",alpha=1)
plt.title("Finished Data")
plt.show()