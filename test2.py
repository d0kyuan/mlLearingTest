#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: test2.py
# Project: learing
# Created Date: Wednesday, June 17th 2020, 2:53:48 pm
# Author: Ray
# -----
# Last Modified: Friday, June 19th 2020, 2:44:56 pm
# Modified By: Ray
# -----
# Copyright (c) 2020 Ray
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import joblib
import h5py
import time
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# register_matplotlib_converters()
# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
# column_names = ['lable', 'x', 'y', 'z', 'x2', 'y2', 'z2', 'timestamp']
# NOTE:訓練模組用的資料
df = pd.read_csv('Target')
df.dropna(axis=0, how='any', inplace=True)

# NOTE:要分類的假資料
df2 = pd.DataFrame(
    {
        'lable': ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        'x': [-1, 1, 2, -1, -1, 1, 2, -1, -1, 1, 2, -1, ],
        'y': [-1, 1, 2, -1, -1, 1, 2, -1, -1, 1, 2, -1, ],
        'z': [-1, 1, 2, -1, -1, 1, 2, -1, -1, 1, 2, -1, ],
        'timestamp': [time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), time.time(), ]
    }, columns=['lable', 'x', 'y', 'z', 'timestamp'])

sns.countplot(x='lable',
              data=df,
              order=df.lable.value_counts().index)

df_train = df
df_test = df2

TIME_STEPS = 10
STEP = 1


def plot_activity(activity, df):
    # print("df['lable']", df['lable'])
    data = df[df['lable'] == activity][['x', 'y', 'z']]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


plot_activity(1, df)
plot_activity(2, df)
plot_activity(3, df)

scale_columns = ['x', 'y', 'z']

scaler = RobustScaler()

scaler = scaler.fit(df_train[scale_columns])

df_train.loc[:, scale_columns] = scaler.transform(
    df_train[scale_columns].to_numpy())
df_test.loc[:, scale_columns] = scaler.transform(
    df_test[scale_columns].to_numpy())


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    # print("Xs", Xs)
    return np.array(Xs), np.array(ys).reshape(-1, 1)


X_train, y_train = create_dataset(
    df_train[['x', 'y', 'z']],
    df_train.lable,
)

X_test, y_test = create_dataset(
    df_test[['x', 'y', 'z']],
    df_test.lable,
)
# print("X_train.shape", X_train.shape)
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
# print("y_train", y_train)
enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
f = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# print(y_pred)


def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()  # ta-da!


plot_cm(
    enc.inverse_transform(y_test),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
)
# print(model.summary())

# model.save('traing_model.h5')
# model.save_weights('traing_model2.h5')
# joblib.dump(scaler, "scaler.joblib")
# model.save('traing_model3.json', save_format='json')
#NOTE: TESTDATA
