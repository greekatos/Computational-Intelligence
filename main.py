import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from IPython.display import display



dftrain = pd.read_csv(r"D:\train\mnist_train.csv")
dftest = pd.read_csv(r"D:\train\test.csv")
#numbers = dftrain.pop('label')

Scaler = StandardScaler()

#standarization
#standarsized = Scaler.fit_transform(dftrain)
#standarsized_test = Scaler.fit_transform(dftest)
#inverse = Scaler.inverse_transform(standarsized)
#inverse_test = Scaler.inverse_transform(standarsized_test)


dftrain_1 = np.array(dftrain)

kfold = KFold(n_splits=5, shuffle = False)
print(kfold)
for train,train in kfold.split(dftrain_1):
    print('train: %s' % (dftrain_1[train]))


#print(dftrain_1)
#scaler = preprocessing.StandardScaler().fit(dftrain)



#print(standarsized)
#print(standarsized_test)
#X_scaled = scaler.transform(dftrain)


