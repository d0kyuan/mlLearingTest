#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: test.py
# Project: learing
# Created Date: Wednesday, June 17th 2020, 2:11:19 pm
# Author: Ray
# -----
# Last Modified: Wednesday, June 17th 2020, 2:53:37 pm
# Modified By: Ray
# -----
# Copyright (c) 2020 Ray
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import numpy
import pandas
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
seed = 7
numpy.random.seed(seed)

df = pandas.read_csv('Target')
dataset = df.values
X = dataset[:, 2:5].astype(float)

Y = list(df['lable'])
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
transformer = encoder
# transformer.fit(X)
# X = transformer.transform(X)
# estimator = KerasClassifier(build_fn=baseline_model,
#                             epochs=200, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
RANDOM_SEED = 42
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED)
model = Sequential()
model.add(Dense(units=64, activation="relu",
                input_shape=[X_train.shape[1]]))
model.add(Dropout(rate=0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(rate=0.5))

model.add(Dense(1))

model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss='mse',
    metrics=['mse'])

BATCH_SIZE = 32

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_mse',
    mode="min",
    patience=10
)

history = model.fit(
    x=X_train,
    y=y_train,
    shuffle=True,
    epochs=100,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)


def plot_mse(history):
    hist = pandas.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train MSE')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val MSE')
    plt.legend()
    plt.show()


# plot_mse(history)
joblib.dump(encoder, "data_transformer.joblib")
model.save("price_prediction_model.h5")
model.load()
