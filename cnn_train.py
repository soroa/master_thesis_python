# set the matplotlib backend so figures can be saved in the background
import argparse
import datetime
import os

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from utils import *

# leave this line

now = datetime.datetime.now()
start_time = now.strftime("%Y-%m-%d %H:%M")

config = yaml_loader("./config_cnn.yaml")
partipant_to_exercises_codes_map = yaml_loader("./config_cnn.yaml").get('participant_to_ex_code_map')
gpus = config.get("cnn_params")['gpus']
nb_classes = 10
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", help="Comma separated GPUS to run the algo on")
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Activation, Flatten, Dropout, Dense, Convolution2D
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model

from data_loading import get_grouped_windows_for_exerices

from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# tf_config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

test_accuracy_over_N = []
train_accuracy_over_N = []


def init_report():
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write(str(config.get("cnn_params")) + "\n")
    f.write(str(config.get("sensor_positions")) + "\n")
    f.write(str(config.get("sensors")) + "\n")
    f.write(str(config.get("data_params")) + "\n")
    f.close()


init_report()


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def write_results_to_report():
    test_a = np.asarray(test_accuracy_over_N)
    train_a = np.asarray(train_accuracy_over_N)
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write(str(get_statistics(test_a)) + "\n")
    f.write(str(get_statistics(train_a)) + "\n")
    f.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + "_" + start_time + ".png")
    plt.clf()


def get_statistics(a):
    max = np.max(a)
    min = np.min(a)
    std = np.std(a)
    avg = np.mean(a)
    return {"max": round(max, 2), "min": round(min, 2), "std": round(std, 2), "avg": round(avg, 2)}


class AccuracyHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.final_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

    def on_train_end(self, logs={}):
        # f = open("./reports/report_" + start_time + ".txt", "a+")
        # f.write("Val Accuracy:  %s\r\n" % self.val_acc[len(self.val_acc) - 1])
        # f.write("Accuracy:  %s\r\n" % self.acc[len(self.acc) - 1])
        # f.close()
        train_accuracy_over_N.append(self.acc[len(self.acc) - 1])
        test_accuracy_over_N.append(self.val_acc[len(self.val_acc) - 1])


history = AccuracyHistory()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                               min_delta=0.001,
                                               patience=3,
                                               verbose=0,
                                               mode='auto', baseline=None)


def model_II(input_shape):
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
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def model_I(input_shape, conv_layer_neurons=10, dropout_rate=0.4, inner_dense_layer_neurons=200):
    model = Sequential()
    model.add(
        Convolution2D(filters=conv_layer_neurons, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                      data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(inner_dense_layer_neurons))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # plot_model(model, show_shapes=True, to_file='model.png')
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def grid_search(model, X, Y, groups):
    # define params
    Y = np_utils.to_categorical(Y - 1, nb_classes)
    conv_layer_neurons = [50, 20, 10, 5]
    dropout_rate = [0.6]
    window_length = [1000, 2000, 3000, 4000]
    inner_dense_layer_neurons = [50, 100, 200, 300]
    input_shape = [(X.shape[1], X.shape[2], 1)]
    # param_grid = dict(input_shape = input_shape, epochs=[10, 20, 30, 30], batch_size=[2, 4, 8, 16, 32])
    param_grid = dict(input_shape=input_shape, dropout_rate=dropout_rate, epochs=[10], batch_size=[16],
                      conv_layer_neurons=conv_layer_neurons)
    model = KerasClassifier(build_fn=model, verbose=0)
    logo = LeaveOneGroupOut()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, verbose=1)
    grid_result = grid.fit(X, Y, groups=groups)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def train(X_train, y_train, X_test, y_test, left_out_participant="", conf_matrix=None):
    y_train = np_utils.to_categorical(y_train - 1, nb_classes)
    y_valid = np_utils.to_categorical(y_test - 1, nb_classes)
    # check to see if we are compiling using just a single GPU
    if len(gpus) <= 1:
        print("[INFO] training with 1 GPU...")
        model = model_I((X_train.shape[1], X_train.shape[2], 1))

    # otherwise, we are compiling using multiple GPUs
    else:
        print("[INFO] training with {} GPUs...".format(gpus))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = model_I((X_train.shape[1], X_train.shape[2], 1))

        # make the model parallel
        model = multi_gpu_model(model, gpus=len(gpus))

    model.fit(X_train, y_train, epochs=config.get("cnn_params")['epochs'], validation_data=(X_test, y_valid),
              batch_size=config.get("cnn_params")['batch_size'],
              # callbacks=[history, early_stopping])
              callbacks=[history])
    if len(gpus) > 1:
        prediction_classes = model.predict(X_test)
        prediction_classes = np.argmax(prediction_classes, axis=1)
    else:
        prediction_classes = model.predict_classes(X_test)
    c = [np.where(r == 1)[0][0] for r in y_valid]
    cm = confusion_matrix(c, prediction_classes)
    if conf_matrix is not None:
        if cm.shape != (10, 10):
            padded_conf_matrix = np.zeros((10, 10)).astype(np.int32)
            padded_conf_matrix[0:cm.shape[0], 0:cm.shape[1]] = cm
            cm = padded_conf_matrix
        conf_matrix = conf_matrix + cm
    else:
        if cm.shape != (10, 10):
            padded_conf_matrix = np.zeros((10, 10)).astype(np.int32)
            padded_conf_matrix[0:cm.shape[0], 0:cm.shape[1]] = cm
            cm = padded_conf_matrix
        conf_matrix = cm

    classes = ["Push ups", "Pull ups", "Burpess", "Deadlifts", "Box jumps", "Squats", "Situps", "WB", "KB Press",
               "Thrusters"]
    if config.get('save_confusion_matrices'):
        plot_confusion_matrix(cm, classes, title=left_out_participant)
    return conf_matrix


if __name__ == "__main__":
    # cm = None
    # for p_test in partipant_to_exercises_codes_map.keys():  # type: object
    #     print("Leaving out " + p_test)
    #     train_exercise_codes = []
    #     for p in partipant_to_exercises_codes_map:
    #         if p == p_test:
    #             continue
    #         train_exercise_codes += (partipant_to_exercises_codes_map[p])
    #
    #     X_train, y_train, X_test, y_test = get_windowed_exerices_feautres_for_training_data(False, train_exercise_codes,
    #                                                                                         config)
    #
    #     # f = open("./reports/report_" + start_time + ".txt", "a+")
    #     # f.write("Leave out:  %s\r\n" % p_test)
    #     # f.close()
    #     cm = train(X_train, y_train, X_test, y_test, p_test, cm)
    #
    # if config.get('save_confusion_matrices'):
    #     plot_confusion_matrix(cm,
    #                           ["Push ups", "Pull ups", "Burpess", "Deadlifts", "Box jumps", "Squats", "Situps", "WB",
    #                            "KB Press",
    #                            "Thrusters"], title="conf_matrix_total")
    #     np.save("confusion_matrix", confusion_matrix)
    # write_results_to_report()

    X, Y, groups = get_grouped_windows_for_exerices(False, config)
    grid_search(model_I, X, Y, groups)
