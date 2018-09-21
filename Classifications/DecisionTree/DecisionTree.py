import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("voice.csv")

male=data[data.label == "male"]
female=data[data.label == "female"]

plt.scatter(male.meanfun,male.sfm,color="blue",label="male",alpha=0.7)
plt.scatter(female.meanfun,female.sfm,color="red",label="female",alpha=0.35)
plt.xlabel("meanfun")
plt.ylabel("sfm")
plt.legend()
plt.show()

y=pd.DataFrame([(1 if i=="male" else 0) for i in pd.DataFrame(data["label"]).values])
x=data.drop(["label"],axis=1)

x,x_test,y,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

x=(x-np.min(x))/(np.max(x)-np.min(x))
score=[]
dt=DecisionTreeClassifier(random_state=5)
dt.fit(x,y)
print("Accuracy=",dt.score(x_test,y_test))