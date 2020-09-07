"""Test pairwise causal Discovery models."""

import os
import pandas as pd
import networkx as nx
from cdt.causality.pairwise import (NCC, IGCI, BivariateFit, CDS,
                                    NCC, RCC, RECI, GNN, Jarfo)
from cdt.independence.graph import Glasso
from cdt.utils.io import read_causal_pairs
from cdt import SETTINGS
from cdt.data import load_dataset
import sys
import time

SETTINGS.NJOBS = 1
import logging
import numpy as np
import random
from sklearn.metrics import roc_curve, auc,accuracy_score,roc_auc_score
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.special import expit





def process_data(prediction, method = "sigmoid"):
    if method == "linear":
        return ( prediction + abs(min(prediction)) ) / max(prediction)
    if method == "sigmoid":
        return expit(prediction)

def read_data(file):
    try:
        df = pd.read_csv(file)
    except pd.io.common.EmptyDataError:
        df = pd.DataFrame()
    return df


def load_data(data_path,label_path =None, test= "01"):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    xxdata_list = []
    yydata_list = []
    if label_path != None:
        ydata_org = read_data(label_path).set_index("SampleID")
        xdata_org = read_causal_pairs(data_path)
        if test == "01":
            ydata_org["Target"].loc[ydata_org["Target"] == -1] = 1
            ydata_org["Target"].loc[ydata_org["Target"] == 0] = -1
            xdata = xdata_org
            ydata = ydata_org
        elif test == "-11":
            ydata = ydata_org.loc[(ydata_org["Target"] == 1) | (ydata_org["Target"] == -1)]
            xdata = xdata_org.loc[(ydata_org["Target"] == 1) | (ydata_org["Target"] == -1)]
        xxdata_list.append(xdata)
        yydata_list.append(ydata)
        return ((pd.concat(xxdata_list), pd.concat(yydata_list)))

    else:
        xdata = read_causal_pairs(data_path)
        xxdata_list.append(xdata)
        return pd.concat(xxdata_list)


def test_pairwise(x_train, y_train, x_test,  method):
    for method in [read_method(method)]:  # , IGCI, BivariateFit, CDS, RCC, NCC, RECI, Jarfo]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(x_train, y_train)
        r = m.predict(x_test)
        r = np.array(r)
    return r.reshape(r.shape[0],1)


def read_method(method):
    if method == "IGCI":
        return IGCI
    if method == "NCC":
        return NCC
    if method == "RCC":
        return RCC
    if method == "BivariateFit":
        return BivariateFit
    if method == "RECI":
        return RECI
    if method == "Jarfo":
        return Jarfo


method = sys.argv[1]

t0 = time.time()
train_data_path = "example/Kaggle_train_data.csv"
train_label_path = "example/Kaggle_train_label.csv"
valid_data_path = "example/Kaggle_valid_data.csv"
test_data_path = "example/Kaggle_test_data.csv"

(x_train, y_train) = load_data(train_data_path,train_label_path,"01")
x_test = load_data(valid_data_path)
y_predict_01 = test_pairwise(x_train, y_train, x_test, method = method)


(x_train, y_train) = load_data(train_data_path,train_label_path,"-11")
x_test = load_data(valid_data_path)
y_predict_11 = test_pairwise(x_train, y_train, x_test, method = method)

if method == "NCC":
    y_predict_01 = process_data(y_predict_01)
    y_predict_11 = process_data(y_predict_11)

pred = y_predict_01 * (2 * y_predict_11- 1)

from collections import defaultdict
y_train =  defaultdict(list)
for i in range(pred.shape[0]):
    label= "valid" + str(i+1)
    y_train['SampleID'].append(label)
    y_train['Target'].append(pred[i][0])

dfy = pd.DataFrame(y_train)
dfy.to_csv("example/valid_result_" + method + ".csv", index=False,header=False)

t1 = time.time()
print(t1 - t0)



(x_train, y_train) = load_data(train_data_path,train_label_path,"01")
x_test = load_data(test_data_path)
y_predict_01 = test_pairwise(x_train, y_train, x_test, method = method)


(x_train, y_train) = load_data(train_data_path,train_label_path,"-11")
x_test = load_data(test_data_path)
y_predict_11 = test_pairwise(x_train, y_train, x_test, method = method)

if method == "NCC":
    y_predict_01 = process_data(y_predict_01)
    y_predict_11 = process_data(y_predict_11)

pred = y_predict_01 * (2 * y_predict_11- 1)

y_train =  defaultdict(list)
for i in range(pred.shape[0]):
    label= "test" + str(i+1)
    y_train['SampleID'].append(label)
    y_train['Target'].append(pred[i][0])

dfy = pd.DataFrame(y_train)
dfy.to_csv("example/test_result_" + method + ".csv", index=False,header=False)

t1 = time.time()
print(t1 - t0)

