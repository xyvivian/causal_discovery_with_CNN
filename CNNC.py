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




def read_data(file):
    try:
        df = pd.read_csv(file)
    except pd.io.common.EmptyDataError:
        df = pd.DataFrame()
    return df



def load_data(data_path,label_path,size, ):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    data_train = []
    target_train = []
    if label_path != None:
        ydata_org = read_data(label_path).set_index("SampleID")
        xdata_org = read_causal_pairs(data_path)

        num_rows = len(xdata_org.index)
        for i in range(num_rows):
            x = xdata_org["A"][i]
            y = xdata_org["B"][i]
            H_T = np.histogram2d(x=x, y=y, bins=size)
            #log_H_T = np.histogram2d(x=np.log10(x + min(x) + 10**-8), y=np.log10(y + min(y) + 10**-8), bins=size)
            H = H_T[0].T
            HT = H / H.max()
            HTX = HT / HT.max()

            hislog = np.log10(HTX + 10 ** -8)

            #log_H = log_H_T[0].T
            #logHT = log_H / log_H.max()
            #logHTX = logHT / logHT.max()

            data_train.append(hislog)
            target_symbol =  ydata_org["Target"][i]
            target_train.append(target_symbol)

        xx = np.array(data_train)[:, :, :, np.newaxis]
        return xx,target_train


def CNN(result_dir, x_train,y_train,x_test,y_test = None, num_class=3):
    if num_class > 2:
        y_train = keras.utils.to_categorical(y_train, num_classes=3, dtype='float32')
        if y_test != None:
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
    model.add(Conv2D(16, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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

    history = model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.33, callbacks=callbacks_list)

    # Save model and weights

    model_path = os.path.join(result_dir + model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.

    if y_test != None:
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        y_predict = model.predict(x_test)
        np.save(save_dir + '/end_y_test.npy', y_test)
        np.save(save_dir + '/end_y_predict.npy', y_predict)




size= int(sys.argv[1])
train_data_path = sys.argv[2]
train_label_path = sys.argv[3]
test_data_path = sys.argv[4]
test_label_path = sys.argv[5]
save_dir = sys.argv[6]

(x_train, y_train) = load_data(train_data_path,train_label_path,size)
(x_test,y_test) = load_data(test_data_path,test_label_path)
CNN(save_dir, x_train, y_train,x_test,y_test, num_class=3)


