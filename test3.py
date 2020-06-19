#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: test3.py
# Project: learing
# Created Date: Wednesday, June 17th 2020, 3:41:10 pm
# Author: Ray
# -----
# Last Modified: Thursday, June 18th 2020, 10:48:21 am
# Modified By: Ray
# -----
# Copyright (c) 2020 Ray
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np
import orjson as json
from scipy import stats
import tensorflow as tf
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib
import h5py
import keras
import pandas as pd
import time
from pylab import rcParams
import requests
from sklearn.model_selection import train_test_split


def _get_tag_info_data(target=None):
    try:
        if target == None:
            r = requests.get(
                'http://192.168.50.253:8080/qpe/getTagInfo?version=2&humanReadable=true&maxAge=10000&tag=d30524000128,2100193100ad')
        else:
            r = requests.get(
                'http://192.168.50.253:8080/qpe/getTagInfo?version=2&humanReadable=true&maxAge=10000&tag=d30524000128,2100193100ad')
        if r.status_code == 200:
            output = []
            # if not tag_data:
            #     return
            # for k, v in tag_data.items():
            #     output.append(_serialize_tag_position(k, v))
            try:
                data = json.loads(r.text)
                return data
                # print("mqtt finish send out time ",datetime.datetime.now())
            except Exception as e:
                pass

        else:
            return {}
    except Exception as e:
        pass


rcParams['figure.figsize'] = 22, 10
# scaler.predict()
# model = keras.Sequential()
# model.load_weights('traing_model2.h5')
classifierLoad = tf.keras.models.load_model('traing_model.h5')
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
scale_columns = ['x', 'y', 'z', 'x2', 'y2', 'z2']
# classifierLoad.train
t = _get_tag_info_data(target=None)['tags']
data = t[0]
data2 = t[1]
df = pd.DataFrame(
    {'lable': ["0"], 'x': [data['acceleration'][0]], 'y': [data['acceleration'][1]], 'z': [data['acceleration'][2]], 'x2': [data2['acceleration'][0]], 'y2': [data2['acceleration'][1]], 'z2': [data2['acceleration'][2]], 'timestamp': [time.time()]}, columns=['lable', 'x', 'y', 'z',  'x2', 'y2', 'z2', 'timestamp'])
df.dropna(axis=0, how='any', inplace=True)
# df.z2.replace(regex=True, inplace=True, to_replace=r';', value=r'')
# df['z2'] = df.z2.astype(np.float64)
# df.dropna(axis=0, how='any', inplace=True)
# print(df.lable.values)
# print(df.head())

# sns.countplot(x='lable',
#               data=df,
#               order=df.lable.value_counts().index)

df_train = df
df_test = df

TIME_STEPS = 200
STEP = 40


scaler = RobustScaler()

scaler = scaler.fit(df_train[scale_columns])
df_train.loc[:, scale_columns] = scaler.transform(
    df_train[scale_columns].to_numpy())
df_test.loc[:, scale_columns] = scaler.transform(
    df_test[scale_columns].to_numpy())

print("df_train.lable", df_train.lable)


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    print("len(X)", len(X))
    for i in range(0, len(X) - time_steps, step):
        print("a")
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# X_train, y_train = create_dataset(
#     df_train[['x', 'y', 'z', 'x2', 'y2', 'z2']],
#     df_train.lable,
#     0,
#     1
# )

# X_test, y_test = create_dataset(
#     df_test[['x', 'y', 'z', 'x2', 'y2', 'z2']],
#     df_test.lable,
#     0,
#     1
# )
print("df_train[['x', 'y', 'z', 'x2', 'y2', 'z2']]",
      df_train[['x', 'y', 'z', 'x2', 'y2', 'z2']])
X_train, X_test, y_train, y_test = train_test_split(
    [data['acceleration'][0], data['acceleration'][1], data['acceleration'][2], data2['acceleration'][0], data2['acceleration'][1], data2['acceleration'][2]],  ['x', 'y', 'z', 'x2', 'y2', 'z2'], test_size=0.3, random_state=0)
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
# print("y_train", y_train)
# print("y_test", y_test)
# enc = enc.fit(y_train)
# # enc.
# y_train = enc.transform(y_train)
# y_test = enc.transform(y_test)
# history = classifierLoad.fit(
#     X_train, y_train,
#     epochs=20,
#     batch_size=64,
#     validation_split=0.1,
#     shuffle=True
# )
# f = classifierLoad.evaluate(X_test, y_test)
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

# enc = enc.fit(y_train)

# y_train = enc.transform(y_train)
# y_test = enc.transform(y_test)
print("X_train", X_train)
y_pred = classifierLoad.predict_classes(X_train)
print(y_pred)
