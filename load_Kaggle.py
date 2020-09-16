import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import random

from collections import defaultdict
import json, re,os, sys

overall_count = 0
x_y_train = defaultdict(list)
y_train = defaultdict(list)
x_y_valid = defaultdict(list)
x_y_test = defaultdict(list)

target_symbol = pd.read_csv("example/Kaggle/SUP1data_split/CEdata_train_target.csv")[
    "Target [1 for A->B; -1 otherwise]"]
for i in range(5998):
    a = np.genfromtxt("example/Kaggle/SUP1data_split/train" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_train['SampleID'].append('pair' + str(overall_count))
    x_y_train['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_train['B'].append(" " + " ".join([str(ii) for ii in y]))

    y_train['SampleID'].append('pair' + str(overall_count))
    y_train['Target'].append(label)
    overall_count += 1


target_symbol = pd.read_csv("example/Kaggle/SUP2data_split/CEdata_train_target.csv")[
    "Target [1 for A->B; -1 otherwise]"]
for i in range(5989):
    a = np.genfromtxt("example/Kaggle/SUP2data_split/train" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_train['SampleID'].append('pair' + str(overall_count))
    x_y_train['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_train['B'].append(" " + " ".join([str(ii) for ii in y]))

    y_train['SampleID'].append('pair' + str(overall_count))
    y_train['Target'].append(label)
    overall_count += 1


target_symbol = pd.read_csv("example/Kaggle/SUP3data_split/CEdata_train_target.csv")[
    "Target [1 for A->B; -1 for B->A and 0otherwise]"]
for i in range(162):
    a = np.genfromtxt("example/Kaggle/SUP3data_split/train" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_train['SampleID'].append('pair' + str(overall_count))
    x_y_train['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_train['B'].append(" " + " ".join([str(ii) for ii in y]))

    y_train['SampleID'].append('pair' + str(overall_count))
    y_train['Target'].append(label)
    overall_count += 1


target_symbol = pd.read_csv("example/Kaggle/CEfinal_train_split/CEfinal_train_target.csv")[
    "Target [1 for A->B; -1 for B->A and 0otherwise]"]
for i in range(4050):
    a = np.genfromtxt("example/Kaggle/CEfinal_train_split/train" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_train['SampleID'].append('pair' + str(overall_count))
    x_y_train['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_train['B'].append(" " + " ".join([str(ii) for ii in y]))

    y_train['SampleID'].append('pair' + str(overall_count))
    y_train['Target'].append(label)
    overall_count += 1

count = 0
for i in range(4050):
    a = np.genfromtxt("example/Kaggle/CEfinal_valid_split/valid" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_valid['SampleID'].append('pair' + str(count))
    x_y_valid['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_valid['B'].append(" " + " ".join([str(ii) for ii in y]))


    count += 1


count = 0
for i in range(4050):
    a = np.genfromtxt("example/Kaggle/CEfinal_test_split/test" + str(i+1) + ".txt")
    x = a[:, 0]
    y = a[:, 1]
    label = target_symbol[i]
    x_y_test['SampleID'].append('pair' + str(count))
    x_y_test['A'].append(" " + " ".join([str(ii) for ii in x]))
    x_y_test['B'].append(" " + " ".join([str(ii) for ii in y]))


    count += 1


dfx = pd.DataFrame(x_y_train)
dfy = pd.DataFrame(y_train)
dfx.to_csv("example/Kaggle_train_data.csv", index=False)
dfy.to_csv("example/Kaggle_train_label.csv", index=False)

dfx = pd.DataFrame(x_y_valid)
dfx.to_csv("example/Kaggle_valid_data.csv", index=False)
dfx = pd.DataFrame(x_y_test)
dfx.to_csv("example/Kaggle_test_data.csv", index=False)

