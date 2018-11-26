import numpy as np
import pandas as pd

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model

from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import math

np.random.seed(7)

msft_dataset = pd.read_csv('./data/UBS.csv')
msft_dataset.head()

msft_dataset['Date'] = pd.to_datetime(msft_dataset['Date'])
msft_dataset['Close'] = pd.to_numeric(msft_dataset['Close'], downcast='float')
msft_dataset.set_index('Date',inplace=True)

msft_close = msft_dataset['Close']
msft_close = msft_close.values.reshape(len(msft_close), 1)
plt.plot(msft_close)
plt.show()

print(msft_close.shape)
scaler = MinMaxScaler(feature_range=(0,1))
msft_close  = scaler.fit_transform(msft_close)
msftTrain , msftTest = msft_close[0:350], msft_close[350:]

def scaler(path):
    msft_dataset = pd.read_csv(path)
    msft_dataset['Date'] = pd.to_datetime(msft_dataset['Data'])
    msft_dataset['Close'] = pd.to_numeric(msft_dataset['Close'], downcast='float')
    msft_dataset.set_index('Date', inplace=True)
    msft_close = msft_dataset['Close']
    msft_close = msft_close.values.reshape(len(msft_close), 1)
    plt.plot(msft_close)
    plt.show()
    scaler = MinMaxScaler(feature_range=(0, 1))
    msft_close = scaler.fit_transform(msft_close)
    msftTrain, msftTest = msft_close[0:350], msft_close[350:]

    return msftTrain, msftTest


def create_ts(ds, series):
    X, Y =[], []
    for i in range(len(ds)-series - 1):
        item = ds[i:(i+series), 0]
        X.append(item)
        Y.append(ds[i+series, 0])
    return np.array(X), np.array(Y)

series = 7

trainX, trainY = create_ts(msftTrain, series=7)
testX, testY = create_ts(msftTest, series=7)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(256, input_shape=(series, 1)))
model.add(Dropout(0.05))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(trainX, trainY, epochs=10, batch_size=32)

model.save('trained_model.h5')

print(trainX.shape)


trainPredictions = model.predict(trainX)
testPredictions = model.predict(testX)
#unscale predictions
trainPredictions = scaler.inverse_transform(trainPredictions)
testPredictions = scaler.inverse_transform(testPredictions)
trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredictions[:, 0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredictions[:, 0]))
print('Train score: %.2f rmse', trainScore)
print('Test score: %.2f rmse', testScore)
'''
trainY, testY = trainY[0], testY[0]
trainY, testY = np.reshape(trainY, (292, 1)), np.reshape(testY, (196, 1))
trainY, testY = scaler.fit_transform(trainY), scaler.fit_transform(testY)
trainY, testY = np.reshape(trainY, (292, )), np.reshape(testY, (196, ))
'''
train_plot = np.empty_like(msft_close)
train_plot[:,:] = np.nan
train_plot[series:len(trainPredictions)+series, :] = trainPredictions

test_plot = np.empty_like(msft_close)
test_plot[:,:] = np.nan
test_plot[len(trainPredictions)+(series*2)+1:len(msft_close)-1, :] = testPredictions

plt.plot(scaler.inverse_transform(msft_close))
plt.plot(train_plot)
plt.plot(test_plot)

plt.show()