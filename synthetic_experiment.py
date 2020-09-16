import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import random
from collections import defaultdict


def reverse_v(param, n,alpha,beta,gamma,number,repeat_time=1):
    z = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    mean_l = param["mean_l"]
    mean_r = param["mean_r"]
    variance_l = param["variance_l"]
    variance_r = param["variance_r"]

    z_tp = np.zeros(n * repeat_time)
    x_tp = np.zeros(n * repeat_time)
    y_tp = np.zeros(n * repeat_time)

    np.random.seed(number * 5 + 1)
    meanx = random.uniform(mean_l, mean_r)
    np.random.seed(number * 5 + 2)
    meany = random.uniform(mean_l, mean_r)
    np.random.seed(number * 5 + 3)
    meanz = random.uniform(mean_l, mean_r)

    np.random.seed(number * 5 + 4)
    variancex = random.uniform(variance_l, variance_r)
    np.random.seed(number * 5 + 5)
    variancey = random.uniform(variance_l, variance_r)
    np.random.seed(number * 5 + 6)
    variancez = random.uniform(variance_l, variance_r)

    epsz = np.power(np.random.normal(loc = meanz, scale = variancez,size=n * repeat_time), 3)

    epsx = np.power(np.random.normal(loc = meanx, scale = variancex, size=n * repeat_time), 3)
    np.random.seed(number * 5 + 3)
    epsy = np.power(np.random.normal(loc = meany, scale = variancey, size=n * repeat_time), 3)

    for i in (range(1,n*repeat_time)):
        x_tp[i] = alpha * x_tp[i - 1] + (1-alpha) * epsx[i]
        y_tp[i] = beta * y_tp[i - 1] + gamma * x_tp[i-1]  + (1- beta - gamma)* epsy[i]
        z_tp[i] = beta * z_tp[i - 1] + gamma * x_tp[i -2] + (1- beta - gamma)*epsz[i]
        if i % repeat_time == 0:
            j = (int)(i / repeat_time)
            z[j] = z_tp[i]
            x[j] = x_tp[i]
            y[j] = y_tp[i]
    return x,y,z

def reverse_v_label():
    label = np.zeros(6)
    label[0] = 1
    label[1] = 1
    label[2] = -1
    label[3] = 0
    label[4] = -1
    label[5] = 0
    return label

def v(param,n,alpha,beta,gamma,number,repeat_time=1):
    z = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    mean_l = param["mean_l"]
    mean_r = param["mean_r"]
    variance_l = param["variance_l"]
    variance_r = param["variance_r"]

    z_tp = np.zeros(n * repeat_time)
    x_tp = np.zeros(n * repeat_time)
    y_tp = np.zeros(n * repeat_time)


    np.random.seed(number * 5 + 1)
    meanx = random.uniform(mean_l, mean_r)
    np.random.seed(number * 5 + 2)
    meany = random.uniform(mean_l, mean_r)
    np.random.seed(number * 5 + 3)
    meanz = random.uniform(mean_l, mean_r)

    np.random.seed(number * 5 + 4)
    variancex = random.uniform(variance_l, variance_r)
    np.random.seed(number * 5 + 5)
    variancey = random.uniform(variance_l, variance_r)
    np.random.seed(number * 5 + 6)
    variancez = random.uniform(variance_l, variance_r)

    epsz = np.power(np.random.normal(loc=meanz, scale=variancez, size=n * repeat_time), 3)

    epsx = np.power(np.random.normal(loc=meanx, scale=variancex, size=n * repeat_time), 3)
    epsy = np.power(np.random.normal(loc=meany, scale=variancey, size=n * repeat_time), 3)

    for i in range(1, n*repeat_time):
        x_tp[i] = alpha * x_tp[i - 1] + (1-alpha) * epsx[i]
        y_tp[i] = alpha * y_tp[i - 1] + (1-alpha) * epsy[i]
        z_tp[i] = alpha * z_tp[i - 1] + beta/2 * x_tp[i - 1] + beta/2 * y_tp[i - 1] + (1- beta - alpha)*epsz[i]
        if i % repeat_time == 0:
            j = (int)(i / repeat_time)
            z[j] = z_tp[i]
            x[j] = x_tp[i]
            y[j] = y_tp[i]
    return x,y,z


def v_label():
    label = np.zeros(6)
    label[0] = 0
    label[1] = 1
    label[2] = 0
    label[3] = 1
    label[4] = -1
    label[5] = -1
    return label


def chain(param,n, alpha, beta, gamma,choice, number, repeat_time=1):
    low = param["mean_l"]
    high = param["mean_r"]


    np.random.seed(number * 5 + 1)
    lowx = random.uniform(low, high)
    np.random.seed(number * 5 + 2)
    highx = random.uniform(low,high)

    np.random.seed(number * 5 + 3)
    lowy = random.uniform(low, high)
    np.random.seed(number * 5 + 4)
    highy = random.uniform(low, high)

    np.random.seed(number * 5 + 5)
    lowz = random.uniform(low, high)
    np.random.seed(number * 5 +6)
    highz = random.uniform(low, high)

    epsx = np.random.uniform(low=lowx, high=highx,size=n*repeat_time )
    epsy = np.random.uniform(low=lowy, high=highy,size=n*repeat_time)
    epsz = np.random.uniform(low=lowz, high=highz,size=n*repeat_time)

    z_tp = np.zeros(n * repeat_time)
    x_tp = np.zeros(n * repeat_time)
    y_tp = np.zeros(n * repeat_time)

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n * repeat_time):
        x_tp[i] = x_tp[i-1]*alpha + (1-alpha)* epsx[i]
        y_tp[i] = y_tp[i-1]*beta + gamma * (x_tp[i-1] -1)**2 +  (1- beta - gamma ) * epsy[i]
        if choice == 0:
            z_tp[i] = beta*z_tp[i-1] + gamma * (y_tp[i-1] -1)**2 + (1- beta - gamma) * epsz[i]
        elif choice == 1:
            z_tp[i] = beta* z_tp[i - 1] + gamma /2 * np.cos(y_tp[i - 1]) + gamma/2 * np.sin(y_tp[i-1]) + (1 - beta - gamma) * epsz[i]

        if i % repeat_time == 0:
            j = (int)(i / repeat_time)
            z[j] = z_tp[i]
            x[j] = x_tp[i]
            y[j] = y_tp[i]
    return x,y,z


def chain_label():
    label = np.zeros(6)
    label[0] = 1
    label[1] = 0
    label[2] = -1
    label[3] = 1
    label[4] = 0
    label[5] = -1
    return label


def generate_data(re_time, param, num, exp,number):
    alpha, beta, gamma, repeat_time = param["alpha"],param["beta"],param["gamma"],param["repeat_time"]
    if exp == "v":
        ret_arr = np.zeros((3,re_time,num))
        for i in range(re_time):
            x,y,z = v(param,n=num,alpha = alpha,beta = beta,gamma = gamma,repeat_time = repeat_time,number=number * 100000 + 5* i )
            ret_arr[0][i] = x
            ret_arr[1][i] = y
            ret_arr[2][i] = z
    elif exp == "reverse_v":
        ret_arr = np.zeros((3,re_time,num))
        for i in range(re_time):
            x,y,z = reverse_v(param,n = num,alpha = alpha,beta = beta,gamma = gamma,repeat_time = repeat_time,number=number * 100000 + 5* i)
            ret_arr[0][i] = x
            ret_arr[1][i] = y
            ret_arr[2][i] = z
    elif exp == "chain":
        ret_arr = np.zeros((3,re_time,num))
        for i in range(re_time):
            x,y,z = chain(param,n = num, alpha = alpha, beta = beta, gamma = gamma,repeat_time = repeat_time, choice=1,number= number * 100000 + 5* i)
            ret_arr[0][i] = x
            ret_arr[1][i] = y
            ret_arr[2][i] = z
    return ret_arr


def generate_labels(exp):
    if exp == "v":
        return v_label()
    if exp == "reverse_v":
        return reverse_v_label()
    if exp == "chain":
        return chain_label()


def generate_causal_pairs(overall_count, data,label):
    x_y_train = defaultdict(list)
    y_train = defaultdict(list)
    for i in tqdm(range(data.shape[1])):
        count = 0
        overall_count += 1
        for j in range(3):
            for k in range(3):
                if j != k:
                    x = data[j][i]
                    y = data[k][i]
                    target_symbol = label[count]

                    x_y_train['SampleID'].append('pair' + str(overall_count))
                    x_y_train['A'].append(" " + " ".join([str(ii) for ii in x]))
                    x_y_train['B'].append(" " + " ".join([str(ii) for ii in y]))

                    y_train['SampleID'].append('pair' + str(overall_count))
                    y_train['Target'].append(target_symbol)
                    count += 1

    dfx = pd.DataFrame(x_y_train)
    dfy = pd.DataFrame(y_train)
    return  dfx, dfy, overall_count



param = {}
alpha = 0.5
beta = 0.5
gamma = 0.5
repeat_time = 1
exp = "chain"
mean_l = 0
mean_r = 10
variance_l = 0
variance_r = 50


param["alpha"] = alpha
param["beta"] = beta
param["gamma"] = gamma
param["repeat_time"] = repeat_time
param["mean_l"] = mean_l
param["mean_r"] = mean_r
param["variance_l"] = variance_l
param["variance_r"] = variance_r

label = generate_labels(exp)
train_ret_arr = generate_data(re_time=8000,num=1000,exp=exp,number=1,param=param)
test_ret_arr  = generate_data(re_time=2000,num=1000,exp=exp,number=2,param=param)
overall_count = 0

dfx,dfy, overall_count = generate_causal_pairs(overall_count,train_ret_arr,label)
dfx.to_csv("example/synthetic_chain_structure_train_data.csv", index=False)
dfy.to_csv("example/synthetic_chain_structure_train_label.csv", index=False)

dfx,dfy, overall_count = generate_causal_pairs(overall_count,test_ret_arr,label)
dfx.to_csv("example/synthetic_chain_structure_test_data.csv", index=False)
dfy.to_csv("example/synthetic_chain_structure_test_label.csv", index=False)


