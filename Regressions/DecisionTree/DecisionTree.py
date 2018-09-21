#%% imported library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor as dt

#%% readed data
df=pd.read_csv("dataset.csv",sep=";")

#%% converted list to DataFrame
dfTemp=pd.DataFrame(df.iloc[:,0].values)
dfRainy=pd.DataFrame(df.iloc[:,1].values)

#%% made Decision Tree
tree=dt()
tree.fit(dfTemp,dfRainy)

#%% border drawing
dfTempArange=np.arange(min(df.iloc[:,0].values)\
                       ,max(df.iloc[:,0].values),0.001).reshape(-1,1)

#%% finded model results
y_results=tree.predict(dfTempArange).reshape(-1,1)

#%% showed values
plt.scatter(dfTemp,dfRainy,color="black")
plt.xlabel("Temperature")
plt.ylabel("Rainy")
plt.plot(dfTempArange,y_results,color="gray")
plt.show()

#%% showed example
print(tree.predict(13))