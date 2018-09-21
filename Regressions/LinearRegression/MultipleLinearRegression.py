#%% imported library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as lr

#%% readed data
df=pd.read_csv("BIST_100.csv",sep=";")

#%% converted ',' to '.'
dfAsels=[float(temp.replace(',','.'))for temp in df.ASELS]
dfUsdTry=[float(temp.replace(',','.'))for temp in df.USDTRY]
dfXU100=[float(temp.replace(',','.'))for temp in df.XU100]

#%% combined datas for Multiple Linear Regression
xLabel={'XU100':dfXU100,'USDTRY':dfUsdTry}

#%% converted list to DataFrame
dfAsels=pd.DataFrame(dfAsels)
xLabel=pd.DataFrame(xLabel)

#%% made Multiple Linear Regression
mLR=lr()
mLR.fit(xLabel,dfAsels)

#%% showed values
plt.scatter(dfUsdTry,dfAsels)
plt.ylabel("Aselsan")
plt.xlabel("Usd-Try")
plt.show()
plt.scatter(dfXU100,dfAsels)
plt.ylabel("Aselsan")
plt.xlabel("XU100")
plt.show()

#%% showed example
print("Accuracy=",mLR.predict(np.array([[83000,2.68]]))[0][0])