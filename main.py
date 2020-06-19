#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: main.py
# Project: learing
# Created Date: Monday, June 15th 2020, 4:07:31 pm
# Author: Ray
# -----
# Last Modified: Friday, June 19th 2020, 2:02:27 pm
# Modified By: Ray
# -----
# Copyright (c) 2020 Ray
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import orjson as json
import requests
import pandas as pd
from sklearn import tree
import time
import joblib
from threading import Thread
import configparser
from sklearn import tree
import datetime
config = configparser.ConfigParser()
config.read('setting.ini')
actionCount = int(config.get('default', 'actionCount'))


def _get_tag_info_data(target=None):
    try:
        if target == None:
            r = requests.get(
                'http://192.168.50.253:8080/qpe/getTagInfo?version=2&humanReadable=true&maxAge=10000')
        else:
            r = requests.get(
                'http://192.168.50.253:8080/qpe/getTagInfo?version=2&humanReadable=true&maxAge=10000&tag='+target)
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


print("開始製作動作偵測訓練模型 ")
tlist = [o['id']for o in _get_tag_info_data()['tags']]
targetNumber = -1
while True:
    print("請先選擇目標 : (輸入數字)")
    for i, o in enumerate(tlist):
        print(f"{i}:{o}")
    number = input()
    try:
        targetNumber = int(number)
        break
    except:
        pass
for i in range(1, actionCount+1):
    actionID = config.get(f'action{i}', 'id')
    actionName = config.get(f'action{i}', 'name')
    actionNeedSecond = int(config.get(f'action{i}', 'second'))
    print(f"3秒後請進行動作{actionID}號的{actionName}動作持續{actionNeedSecond}秒")
    time.sleep(3)
    cars = {'lable': [],
            'x': [],
            'y': [],
            'z': [],
            'x2': [],
            'y2': [],
            'z2': [],
            'timestamp': []
            }
    non_stop = True

    def count_down(second):
        global non_stop
        print(second)
        i = 0
        while not i == second:
            i += 1
            print(f'i {i}s')
            time.sleep(1)
        non_stop = False

    Thread(target=count_down, args=(actionNeedSecond,)).start()

    while(non_stop):
        t = _get_tag_info_data(target=tlist[targetNumber])['tags']
        data = t[0]
        # data2 = t[1]
        # print('data', data, 'data2', data2)
        cars['lable'].append(str(actionID))
        cars['x'].append(data['acceleration'][0])
        cars['y'].append(data['acceleration'][1])
        cars['z'].append(data['acceleration'][2])
        cars['timestamp'].append(time.time())
    # print(cars)
    df = pd.DataFrame(
        cars, columns=['lable', 'x', 'y', 'z', 'timestamp'])
    if i == 1:
        df.to_csv('Target', index=False)
    else:
        df2 = pd.read_csv('Target')
        df3 = df2.append(df, ignore_index=True)
        df3.to_csv('Target', index=False)
df = pd.read_csv('Target')
clf = tree.DecisionTreeClassifier()
t = [[o[0], o[1], o[2]]
     for o in zip(df['x'], df['y'], df['z'])]
labels = list(df['lable'])
clf = clf.fit(t, labels)
joblib.dump(clf, 'clf2.pkl')
print('完成')
# print(df2)
# print(df)
