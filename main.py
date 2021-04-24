import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


dftrain = pd.read_csv(r"D:\train\test.csv")
numbers = dftrain.pop('label')

#scaler = preprocessing.StandardScaler().fit(dftrain)
Scaler = StandardScaler()

standarsized = Scaler.fit_transform(dftrain)
inverse = Scaler.inverse_transform(standarsized)

print(standarsized)
#X_scaled = scaler.transform(dftrain)


