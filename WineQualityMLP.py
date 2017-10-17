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
test_len = len - training_len
import random

random.seed(2017)
random.shuffle(data)

training_data = [x[0].split(";") for x in data[0:training_len]]
test_data = [x[0].split(";") for x in data[training_len:]]

import numpy as np

training_data_d = np.array(training_data)[:, 0:-1].astype(np.float64)
training_data_l = np.array(training_data)[:, -1].astype(np.float64)
training_data_l = np.array([1 if x > 5 else 0 for x in training_data_l]).astype(np.float64)
test_data_d = np.array(test_data)[:, 0:-1].astype(np.float64)
test_data_l = np.array(test_data)[:, -1].astype(np.float64)
test_data_l = np.array([1 if x > 5 else 0 for x in test_data_l]).astype(np.float64)

w1 = np.array([(float)(random.gauss(0,0.001)) for x in range(11)])
w2 = np.array([(float)(random.gauss(0,0.001)) for x in range(11)])
w3 = np.array([(float)(random.gauss(0,0.001)) for x in range(11)])
w4 = np.array([(float)(random.gauss(0,0.001)) for x in range(11)])
w = np.array([(float)(random.gauss(0,0.001)) for x in range(4)])

b1 = (float)(random.gauss(0,0.001))
b2 = (float)(random.gauss(0,0.001))
b3 = (float)(random.gauss(0,0.001))
b4 = (float)(random.gauss(0,0.001))
b = (float)(random.gauss(0,0.001))

import math


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        if x > 0:
            return 1
        else :
            return 0



alpha = 0.001
printnum = 100

for loop in range(10000):

    if loop % printnum == 0:
        cnt = 0
        for i in range(training_len):
            tuple = training_data_d[i]
            lable = training_data_l[i]

            o1 = np.matmul(w1.transpose(), tuple) + b1
            s1 = sigmoid(o1)
            o2 = np.matmul(w2.transpose(), tuple) + b2
            s2 = sigmoid(o2)
            o3 = np.matmul(w1.transpose(), tuple) + b3
            s3 = sigmoid(o3)
            o4 = np.matmul(w2.transpose(), tuple) + b4
            s4 = sigmoid(o4)

            input = np.array(list([s1, s2, s3, s4])).astype(np.float64)
            o = np.matmul(w.transpose(), input) + b
            s = sigmoid(o)

            if s > 0.5 and lable == 1:
                cnt += 1
            if s < 0.5 and lable == 0:
                cnt += 1

        print("training:", cnt / training_len)

    if loop % printnum == 0:
        cnt = 0
        for i in range(test_len):
            tuple = test_data_d[i]
            lable = test_data_l[i]

            o1 = np.matmul(w1.transpose(), tuple) + b1
            s1 = sigmoid(o1)
            o2 = np.matmul(w2.transpose(), tuple) + b2
            s2 = sigmoid(o2)
            o3 = np.matmul(w1.transpose(), tuple) + b3
            s3 = sigmoid(o3)
            o4 = np.matmul(w2.transpose(), tuple) + b4
            s4 = sigmoid(o4)

            input = np.array(list([s1, s2, s3, s4])).astype(np.float64)
            o = np.matmul(w.transpose(), input) + b
            s = sigmoid(o)

            if s > 0.5 and lable == 1:
                cnt += 1
            if s < 0.5 and lable == 0:
                cnt += 1

        print("test:", cnt / test_len)

    cnt = 0
    for i in range(training_len):
        tuple = training_data_d[i]
        lable = training_data_l[i]

        o1 = np.matmul(w1.transpose(), tuple) + b1
        s1 = sigmoid(o1)
        o2 = np.matmul(w2.transpose(), tuple) + b2
        s2 = sigmoid(o2)
        o3 = np.matmul(w1.transpose(), tuple) + b3
        s3 = sigmoid(o3)
        o4 = np.matmul(w2.transpose(), tuple) + b4
        s4 = sigmoid(o4)

        input = np.array(list([s1, s2, s3, s4])).astype(np.float64)
        o = np.matmul(w.transpose(), input) + b
        s = sigmoid(o)

        dEdo = (s - lable) * s * (1 - s)
        delta = np.multiply(dEdo * alpha, input)

        w = np.subtract(w, delta)
        b -= alpha * dEdo

        dEdo1 = s1 * (1 - s1) * w[0] * dEdo
        delta1 = np.multiply(alpha * dEdo1, tuple)
        w1 = np.subtract(w1, delta1)
        b1 -= alpha * dEdo1;

        dEdo2 = s2 * (1 - s2) * w[1] * dEdo
        delta2 = np.multiply(alpha * dEdo2, tuple)
        w2 = np.subtract(w2, delta2)
        b2 -= alpha * dEdo2;

        dEdo3 = s3 * (1 - s3) * w[2] * dEdo
        delta3 = np.multiply(alpha * dEdo3, tuple)
        w3 = np.subtract(w3, delta1)
        b3 -= alpha * dEdo3;

        dEdo4 = s4 * (1 - s4) * w[3] * dEdo
        delta4 = np.multiply(alpha * dEdo4, tuple)
        w4 = np.subtract(w4, delta4)
        b4 -= alpha * dEdo4;


        # for debugging
        if i % printnum ==0:
            o1_ = np.matmul(w1.transpose(), tuple) + b1
            s1_ = sigmoid(o1)
            o2_ = np.matmul(w2.transpose(), tuple) + b2
            s2_ = sigmoid(o2)
            i3_ = np.array(list([s1, s2])).astype(np.float64)
            o3_ = np.matmul(w.transpose(), input) + b
            s3_ = sigmoid(o)
            o1=o1 #nop


