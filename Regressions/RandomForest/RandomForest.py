#%% imported library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rf

#%% readed data
df=pd.read_csv("dataset.csv",sep=";")

#%% converted list to DataFrame
dfTemp=pd.DataFrame(df.iloc[:,0].values)
dfRainy=pd.DataFrame(df.iloc[:,1].values)

#%% showed values
plt.scatter(dfTemp,dfRainy)
plt.xlabel("Temperature")
plt.ylabel("Rainy")
plt.show()

#%% made Decision Tree
randomF=rf(n_estimators=100,random_state=42)
randomF.fit(dfTemp,dfRainy.values.ravel())

#%% showed example
print(randomF.predict(25))