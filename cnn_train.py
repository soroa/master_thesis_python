# set the matplotlib backend so figures can be saved in the background
import argparse
import json
import os


nb_epoch = 3
batch_size = 16
gpus = 0
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", help="Comma separated GPUS to run the algo on")
parser.add_argument("-bs", "--batch_size", help="Batch size", type=int)
parser.add_argument("-e", "--epochs", help="Batch size", type=int)
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpus = len((args.gpus).split(','))

if args.batch_size:
    batch_size = args.batch_size

if args.epochs:
    nb_epoch = args.epochs

import tensorflow as tf
import keras
import numpy as np
from keras import Sequential
from keras.layers import Convolution1D, Activation, Flatten, Dropout, Dense, MaxPooling1D, Convolution2D
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

from data_loading import get_windowed_exerices_feautres_for_training_data


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.final_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

    def on_train_end(self, logs=None):
        self.final_acc.append(logs.get('val_acc'))
#

history = AccuracyHistory()

def ski_jumps_model_II(input_shape):
    # 1 Convo, 1 Dense, 1 Softmax
    model = Sequential()
    model.add(
        Convolution2D(filters=10, kernel_size=(10, 3), strides=(3, 1), input_shape=input_shape,
                      data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(
        Convolution2D(filters=15, kernel_size=(10, 3), strides=(3, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(
        Convolution2D(filters=20, kernel_size=(10, 3), strides=(3, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def ski_jumps_model_I():
    ##3 Convo, 1 Dense, 1 Softmax
    model = Sequential()
    model.add(
        Convolution2D(nb_filter=32, kernel_size=10, input_shape=(X_train.shape[1], X_train.shape[2]), padding="same",
                      strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(
        Convolution1D(nb_filter=32, kernel_size=10, input_shape=(X_train.shape[1], X_train.shape[2]), padding="same",
                      strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


# f = open('destFile.txt', 'r')
# partipant_to_exercises_codes_map = json.loads(f.read())
# train_test_sets = {}
# for p_test in partipant_to_exercises_codes_map.keys():  # type: object
#     print("Leaving out " + p_test)
#     test_exercise_codes = partipant_to_exercises_codes_map[p_test]
#     train_exercise_codes = []
#     for p in partipant_to_exercises_codes_map:
#         if p == p_test:
#             continue
#         train_exercise_codes.append(partipant_to_exercises_codes_map[p])
#     train_test_sets[p_test] = {"test": test_exercise_codes, "train": train_exercise_codes}

print(device_lib.list_local_devices())
X_train, y_train, X_test, y_test = get_windowed_exerices_feautres_for_training_data(False)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

nb_classes = 10

# check to see if we are compiling using just a single GPU
if gpus <= 1:
    print("[INFO] training with 1 GPU...")
    model = ski_jumps_model_II((X_train.shape[1], X_train.shape[2], 1))

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(gpus))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        # initialize the model
        model = ski_jumps_model_II((X_train.shape[1], X_train.shape[2], 1))

    # make the model parallel
    model = multi_gpu_model(model, gpus=gpus)

y_train = np_utils.to_categorical(y_train - 1, nb_classes)
y_valid = np_utils.to_categorical(y_test - 1, nb_classes)

sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_test, y_valid), batch_size=batch_size,  callbacks=[history])
score = model.evaluate(X_test, y_valid, verbose=0)
print(score)
plt.plot(range(0, nb_epoch), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()