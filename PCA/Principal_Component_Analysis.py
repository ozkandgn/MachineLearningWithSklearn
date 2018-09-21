import pandas as pd
from sklearn.decomposition import PCA

data=pd.read_csv("data.csv")
data["class"]=[0 if i=="Iris-setosa" else \
    (1 if i=="Iris-versicolor" else 2) for i in data["class"]]

pca=PCA(n_components=2,whiten=True)

x=data.drop(["class"],axis=1).values

pca.fit(x)
pca_values=pca.transform(x)

print("Variance=",pca.explained_variance_ratio_)
print("Sum=",sum(pca.explained_variance_ratio_))