import numpy as np
import json
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import random
#import matplotlib.pyplot as plt
from collections import defaultdict
import json, re,os, sys
from cdt.utils.io import read_causal_pairs
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import roc_curve, auc
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
model_name = 'keras_cnn_trained_model_shallow.h5'

#sample weights are specified by turbigen experiment paper
sample_weight = [1 / 6,1 / 6,1 / 6,1 / 6,1 / 7,1 / 7,1 / 7,1 / 7,1 / 7,1 / 7,1 / 7,1 / 2,1 / 4,1 / 4,1 / 4,1 / 4,1 / 2,1,1
,1 / 6,1 / 6,1 / 3,1 / 3,1 / 3,1 / 8,1 / 8,1 / 8,1 / 8,1 / 8,1 / 8,1 / 8,1 / 8,1 / 5,1 / 5,1 / 5,1 / 5,1 / 5,1 / 4,1 / 4,1 / 4
,1 / 4,1 / 2,1 / 4,1 / 4,1 / 4,1 / 4,1,1,1 / 3,1 / 3,1 / 3,1 / 12,1 / 12,1 / 12,1 / 12,1 / 12,1 / 12,1 / 12,1 / 12,1 / 3,1 / 3,1 / 3
,1,1,1,1,1 / 12,1 / 12,1 / 12,1,1 / 2,1 / 3,1 / 3,1 / 3,1,1,1,1,1,1 / 4,1 / 4,1 / 4,1 / 4,1,1 / 3,1 / 3,1 / 3,1 / 2,1 / 2
,1,1,1,1,1,1,1]




def return_array(histogramList,LabelList, listObj,size):
    ## return numpy array of training sample and class label
    for obj in listObj:
        tempX = np.array(obj["trainX"])
        tempY = np.array(obj["trainY"])


        H_T = np.histogram2d(x=tempX, y=tempY, bins=size)
        H = H_T[0].T
        HT = H / H.max()
        HTX = HT / HT.max()
        loghis = np.log10(HTX + 10**-8)


        H_T = np.histogram2d(x=tempY, y=tempX, bins=size)
        H = H_T[0].T
        HT = H / H.max()
        HTX_2 = HT / HT.max()
        loghis_2 = np.log10(HTX_2 + 10**-8)

        tempLabel = [obj["label"]]
        LabelList.append(tempLabel)
        LabelList.append([1 - tempLabel[0]])
        histogramList.append(HTX)
        histogramList.append(HTX_2)

def return_array_test(histogramList,LabelList, listObj,size):
    ## return numpy array of training sample and class label
    for obj in listObj:
        tempX = np.array(obj["trainX"])
        tempY = np.array(obj["trainY"])


        H_T = np.histogram2d(x=tempX, y=tempY, bins=size)
        H = H_T[0].T
        HT = H / H.max()
        HTX = HT / HT.max()
        #loghis = np.log10(HTX + 10**-8)

        tempLabel = [obj["label"]]
        LabelList.append(tempLabel)
        histogramList.append(HTX)

def load_test(dataset):
    histogramList = []
    LabelList = []
    with open(dataset, "r") as tubDataReader:
        for line in tubDataReader:
            data = json.loads(line)
            return_array_test(histogramList,LabelList,[data],size=16)

        xx = np.array(histogramList)[:, :, :, np.newaxis]
        label = np.array(LabelList)

    return xx, label



def load(dataset):
    histogramList = []
    LabelList = []
    with open(dataset, "r") as tubDataReader:
        count = 0
        correct = 0
        for line in tubDataReader:
            data = json.loads(line)
            return_array(histogramList,LabelList,[data],size=16)

        xx = np.array(histogramList)[:, :, :, np.newaxis]
        label = np.array(LabelList)

    return xx, label


def CNN(result_dir, x_train,y_train,x_test,y_test = None, num_class=3):
    if num_class > 2:
        y_train = keras.utils.to_categorical(y_train, num_classes=3, dtype='float32')
        if type(y_test) == list or type(y_test) == np.ndarray:
            y_test = keras.utils.to_categorical(y_test, num_classes=3, dtype='float32')

    print(x_train.shape)
    print(y_train.shape)
    print("###################################################")




    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')

    save_dir = os.path.join(result_dir)  ## the result folder


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(16, (3, 3), padding='same', activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))

    if num_class < 2:
        print('no enough categories')
        sys.exit()

    elif num_class == 2:
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    else:
        model.add(Dense(num_class))
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                  verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]

    history = model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.25, callbacks=callbacks_list)

    # Save model and weights

    model_path = os.path.join(result_dir + model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.

    if type(y_test) == list or type(y_test) == np.ndarray:
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        y_predict = model.predict(x_test)
        np.save(save_dir + '/end_y_test.npy', y_test)
        np.save(save_dir + '/end_y_predict.npy', y_predict)

    #scores = model.evaluate(x_tur, y_tur, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
    y_predict = model.predict(x_tur)
    from sklearn.metrics import accuracy_score


    #y_tur_th = keras.utils.to_categorical(y_tur, num_classes=3, dtype='float32')

    prediction = []
    for i in range(y_tur.shape[0]):
        print(y_tur[i], y_predict[i])

    y_predict = model.predict_classes(x_tur)
    print(accuracy_score(y_tur,y_predict, sample_weight=sample_weight))
    np.save(save_dir + '/end_y_tur.npy', y_tur)
    np.save(save_dir + '/end_y_predict_tur.npy', y_predict)


x_data, y_data = load("casual-data-gen-30K.json-original")
x_tur, y_tur = load_test("tubehengenDataFormat.json")
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2, random_state = 42)
CNN("result",X_train,y_train,X_test,y_test , num_class=2)

