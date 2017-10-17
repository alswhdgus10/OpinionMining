def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # header 제외
    return data


data = read_data('rsrc/winequality-red.csv')
training_rate = .7;
test_percentage = .3;
len = len(data)
training_len = int(len * training_rate)
test_len = len-training_len
import random

random.seed(2019)
random.shuffle(data)

training_data = [x[0].split(";") for x in data[0:training_len]]
test_data = [x[0].split(";") for x in data[training_len:]]

import numpy as np

training_data_d = np.array(training_data)[:, 0:-1].astype(np.float64)
training_data_l = np.array(training_data)[:, -1].astype(np.float64)
training_data_l = np.array([1 if x> 5 else 0 for x in training_data_l]).astype(np.float64)
test_data_d = np.array(test_data)[:, 0:-1].astype(np.float64)
test_data_l = np.array(test_data)[:, -1].astype(np.float64)
test_data_l = np.array([1 if x> 5 else 0 for x in test_data_l]).astype(np.float64)

w = np.array([(float)(random.random() - 0.5) for x in range(11)])
b = (float)(random.random() - 0.5)

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


alpha = 0.001
printnum = 100

for loop in range(1000):

    cnt = 0
    for i in range(training_len):
        tuple = training_data_d[i]
        lable = training_data_l[i]
        o = np.matmul(w.transpose(), tuple) + b
        s = sigmoid(o)
        dEdo = (s - lable) * s * (1 - s)

        delta = np.multiply(dEdo * alpha, tuple)
        w = np.subtract(w, delta)
        b -= alpha * dEdo

    for i in range(training_len):
        tuple = training_data_d[i]
        lable = training_data_l[i]
        o = np.matmul(w.transpose(), tuple) + b
        s = sigmoid(o)

        if s>0.5 and lable == 1:
            cnt += 1
        if s<0.5 and lable == 0:
            cnt += 1
    if(loop%printnum==0):
        print("training:", cnt/training_len)

    cnt=0
    for i in range(test_len):
        tuple = test_data_d[i]
        lable = test_data_l[i]
        o = np.matmul(w.transpose(), tuple) + b
        s = sigmoid(o)

        if s > 0.5 and lable == 1:
            cnt += 1
        if s < 0.5 and lable == 0:
            cnt += 1
    if (loop % printnum == 0):
        print("test:", cnt / training_len)




