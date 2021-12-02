import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Read the CSV Data Set
train = pd.read_csv(r"D:\train\mnist_train.csv")
test = pd.read_csv(r"D:\train\test.csv")

numbers = train.pop('label')
numb = test.pop('label')

# Normalize both Data Sets
norm_dataset = StandardScaler().fit_transform(X=train)

norm_datatest = StandardScaler().fit_transform(X=test)

# Inputs for both Train and Test Data
X_tr = norm_dataset[:, :]
X_ts = norm_datatest[:, :]

# Outputs for both Train and Test Data
Y_tr = norm_dataset[:, 0]
Y_ts = norm_datatest[:, 0]

rmseList = []

# Cross Validation with 5 folds
kfold = KFold(n_splits=5, shuffle=True)

for i, (train, test) in enumerate(kfold.split(X_tr)):
    model = Sequential()
    # Input Layer
    model.add(Dense(784,kernel_regularizer=l2(0.9), bias_regularizer=l2(0.9), activation='relu', input_dim=784))#kernel_regularizer=l2(0.5), bias_regularizer=l2(0.5),

    # Second Hidden Layer
    model.add(Dense(1000,kernel_regularizer=l2(0.9), bias_regularizer=l2(0.9), activation='relu'))#kernel_regularizer=l2(0.5), bias_regularizer=l2(0.5),

    # Output Layer
    model.add(Dense(10, activation='softmax'))


    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    model.summary()
    p = keras.optimizers.SGD(lr=0.1, momentum=0.6, decay=0.0, nesterov=False)

    model.compile(loss='mean_squared_error', optimizer=p, metrics=['accuracy'])#mean_squared_error
    fitness = model.fit(X_tr, Y_tr, validation_data=(X_ts, Y_ts), epochs=50, batch_size=500, verbose=1)#sparse_categorical_crossentropy

    scores = model.evaluate(X_ts, Y_ts, verbose=1)

    rmseList.append(scores[0])
    print("Fold :", i, " CE:", scores[0])

    predict = model.predict(X_ts)
    print(scores)

    print('Accuracy: %.2f' % (scores[0] * 100))

print("CE: ", np.mean(rmseList))

history = fitness.history

print(history['loss'])
print(history['accuracy'])
print(history['val_loss'])
print(history['val_accuracy'])

loss = history['loss']
val_loss = history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'])
plt.figure()

accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy', 'val_acuracy'])
plt.show()
