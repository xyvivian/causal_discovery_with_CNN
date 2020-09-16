import pandas as pd
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



def read_causal_pairs(filename):
    """A simplified version of converting the causal pairs, from cdt.io by Diviyan Kalainathan
    Convert a ChaLearn Cause effect pairs challenge format into numpy.ndarray."""
    def convert_row(row):
        a = row["A"].split(" ")
        b = row["B"].split(" ")
        if a[0] == "":
            a.pop(0)
            b.pop(0)
        if a[-1] == "":
            a.pop(-1)
            b.pop(-1)
        a = np.array([float(i) for i in a])
        b = np.array([float(i) for i in b])
        return row['SampleID'], a, b

    data = pd.read_csv(filename)
    conv_data = []
    for _, row in data.iterrows():
        conv_data.append(convert_row(row))
    df = pd.DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    df = df.set_index("SampleID")
    return df


def read_data(file):
    try:
        df = pd.read_csv(file)
    except pd.io.common.EmptyDataError:
        df = pd.DataFrame()
    return df


def load_data(data_path,label_path=None,size=8, log= True):
    """Data loading process
     Args:                data_path: the csv file for x,y causal pairs
                          label_path: the csv file for labels
                          size: the dimension of histogram, for example, 8 indicates a 8x8
                          log: flag for taking the log after the generated histograms
    Return: the generated numpy array of histograms, their labels


     Data should be in the format as the following:
     SampleID, A, B
     pair0, 0.1 0.2 0.5 0.2 ...., 1.2 1.5 1.2 1.1 ...
     For label file, it should also be a csv file that starts with
     SampleID,Target
     pair0,1  pair1,-1, ....
     """
    data_train = []
    target_train = []

    xdata_org = read_causal_pairs(data_path)

    num_rows = len(xdata_org.index)
    for i in range(num_rows):
        x = xdata_org["A"][i]
        y = xdata_org["B"][i]
        H_T = np.histogram2d(x=x, y=y, bins=size)
        H = H_T[0].T
        HT = H / H.max()
        HTX = HT / HT.max()
        if log:
            hislog = np.log10(HTX + 10 ** -8)
        else:
            hislog = HTX
        data_train.append(hislog)

        if label_path != None:
            ydata_org = read_data(label_path).set_index("SampleID")
            target_symbol =  ydata_org["Target"][i]
            target_train.append(target_symbol)

    xx = np.array(data_train)[:, :, :, np.newaxis]
    if label_path != None:
        return xx,target_train
    else:
        return xx


def CNN(result_dir, x_train,y_train,x_test,y_test = None, num_class=3):
# The CNN: for generated histogram classifications, result should be stored as in the result_dir
    if num_class > 2:
        y_train = keras.utils.to_categorical(y_train, num_classes=3, dtype='float32')
        if y_test != None:
            y_test = keras.utils.to_categorical(y_test, num_classes=3, dtype='float32')

    print(x_train.shape)
    print(y_train.shape)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')

    save_dir = os.path.join(result_dir)  ## the result folder


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    #A typical CNN structrure with 3 convolutional layers and one dense layer
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3),padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3),padding='same',
                     input_shape=x_train.shape[1:]))

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

##############################################################################
#params: is_log: flag for taking the log on generated histogram
#size: histogram size For example 8 means 8x8 histogram
#num of classes for CNN prediction, can be both binary or three-way
#save_dir for weights of trained CNN
# train_data_path: csv file for causal pairs A,B
# train_label_path: csv file for labels
# test_data_path: csv file for causal pairs A,B (for prediction)
# test_data_path: optional, for evaluating the accuracy of A,B causal effects

is_log = sys.argv[1].lower() == 'true'
size= int(sys.argv[2])
num_class = int(sys.argv[3])
save_dir = sys.argv[4]
train_data_path = sys.argv[5]
train_label_path = sys.argv[6]
test_data_path = sys.argv[7]


if len(sys.argv) > 8:
    test_label_path = sys.argv[8]
    (x_train, y_train) = load_data(train_data_path,train_label_path,size)
    (x_test,y_test) = load_data(test_data_path,test_label_path)
    CNN(result_dir =save_dir, x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test, num_class=num_class)
else:
    (x_train, y_train) = load_data(train_data_path, train_label_path, size)
    (x_test) = load_data(test_data_path)
    CNN(result_dir=save_dir, x_train=x_train, y_train=y_train, x_test=x_test, num_class=num_class)


