import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as testSplit

data=pd.read_csv("data.csv")
data.drop(["Country","Happiness.Rank"],axis=1,inplace=True)
y=pd.DataFrame(data["Happiness.Score"])
x=data.drop(["Happiness.Score"],axis=1)

x=(x-np.min(x))/(np.max(x)-np.min(x))
y=(y-np.min(y))/(np.max(y)-np.min(y))

x,xTest,y,yTest=testSplit(x,y,test_size=0.2,random_state=42)

x=x.T
y=y.T
xTest=xTest.T
yTest=yTest.T

Create().CreateWeight(4)

class Create():
    def CreateWeight(size):
        weight=np.random.rand(size,1)