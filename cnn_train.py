# set the matplotlib backend so figures can be saved in the background
import argparse
import datetime
import os

# leave this line
import psutil
from keras.applications import ResNet50
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit

from utils import *

# leave this line

now = datetime.datetime.now()
start_time = now.strftime("%Y-%m-%d %H:%M")
nb_classes = 10
config = yaml_loader("./config_cnn.yaml")

gpus = config.get("cnn_params")['gpus']
print(gpus)
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", help="Comma separated GPUS to run the algo on")
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import tensorflow as tf
import keras
from keras import Sequential, Input, Model
from keras.layers import Activation, Flatten, Dropout, Dense, Convolution2D, K, concatenate
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model

from data_loading import get_grouped_windows_for_exerices, get_grouped_windows_for_rep_transistion
from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
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
                                               mode='auto')


def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()


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


def model_I(input_shape,
            conv_layer_1_filters=50, dropout_1=0.5,
            conv_layer_2_filters=100, dropout_2=0.5,
            conv_layer_3_filters=25, dropout_3=0.5,
            conv_layer_4_filters=50, dropout_4=0.5,
            conv_layer_5_filters=25, dropout_5=0.5,
            inner_dense_layer_neurons=250):
    K.clear_session()
    model = Sequential()
    model.add(Convolution2D(filters=conv_layer_1_filters, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_1))
    model.add(Convolution2D(filters=conv_layer_2_filters, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_2))
    model.add(Convolution2D(filters=conv_layer_3_filters, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_3))
    model.add(Convolution2D(filters=conv_layer_4_filters, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_4))
    model.add(Convolution2D(filters=conv_layer_5_filters, kernel_size=(10, 15), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_5))
    model.add(Flatten())
    model.add(Dense(inner_dense_layer_neurons))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # plot_model(model, show_shapes=True, to_file='model.png')
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    mem = get_mem_usage()
    print()
    print()
    print()
    print('mem: {}'.format(mem))
    print()
    print()
    print()
    return model


def model_rep_transistions(input_shape,
                           conv_layer_1_filters=55, dropout_1=0.6,
                           inner_dense_layer_neurons=250, n_classes=nb_classes):
    # sensor data
    conv_input = Input(shape=input_shape)
    conv_output = Convolution2D(filters=conv_layer_1_filters, kernel_size=(10, 15), strides=(3, 1),
                                input_shape=input_shape,
                                border_mode='same',
                                data_format="channels_last")(conv_input)
    act = Activation('relu')(conv_output)
    after_dropout = Dropout(dropout_1)(act)
    flattened = Flatten()(after_dropout)

    label_input = Input(shape=(1,))

    # Merge and add dense layer
    merge_layer = concatenate([label_input, flattened])
    pre_output = Dense(inner_dense_layer_neurons)(merge_layer)
    output = (Dense(n_classes))(pre_output)
    output = Activation('softmax')(output)

    # Define model with two inputs
    model = Model(inputs=[conv_input, label_input], outputs=[output])
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def pad_and_reshape_for_resnet(X):
    X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 6, 3))
    padded_X = np.zeros((X.shape[0], X.shape[1], 198, 3))
    padded_X[:, :, :X_reshaped.shape[2], :] = X_reshaped
    return padded_X


def resNet():
    X, Y, groups = get_grouped_windows_for_exerices(False, config)
    (train, test) = split_train_test(X, Y, groups)
    # X_n = X[:, :, order, :]

    train_reshaped = pad_and_reshape_for_resnet(train[0])
    test_reshaped = pad_and_reshape_for_resnet(test[0])

    resnet = ResNet50(input_shape=(train_reshaped.shape[1], train_reshaped.shape[2], train_reshaped.shape[3]),
                      include_top=False, classes=nb_classes)
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(train_reshaped, train[1], validation_data=(test_reshaped, test[1]))


def grid_search(model, X, Y, groups):
    # define params
    print("start Gridsearch")

    # params
    conv_layer_neurons = [50]
    # dropout_rate = [0.4, 0.6, 0.8]
    # inner_dense_layer_neurons = [100, 250, 500]

    conv_layer_1_filters = [50, 100]
    dropout_1 = [0.5]
    conv_layer_2_filters = [50, 100]
    dropout_2 = [0.5]
    conv_layer_3_filters = [50, 100]
    dropout_3 = [0.5]
    conv_layer_4_filters = [50, 100]
    dropout_4 = [0.5]
    conv_layer_5_filters = [50, 100]
    dropout_5 = [0.5]
    input_shape = [(X.shape[1], X.shape[2], 1)]
    n_conv_layer = [1, 2, 3, 5]
    param_grid = dict(input_shape=input_shape, conv_layer_1_filters=conv_layer_1_filters,
                      dropout_1=dropout_1,
                      conv_layer_2_filters=conv_layer_2_filters,
                      dropout_2=dropout_2,
                      conv_layer_3_filters=conv_layer_3_filters,
                      dropout_3=dropout_3,
                      conv_layer_4_filters=conv_layer_4_filters,
                      dropout_4=dropout_4,
                      conv_layer_5_filters=conv_layer_5_filters,
                      dropout_5=dropout_5,
                      epochs=[5],
                      batch_size=[10])

    model = KerasClassifier(build_fn=model, verbose=1)
    logo = GroupShuffleSplit(test_size=0.2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, verbose=10, n_jobs=1)
    print("Gridsearch fit")
    grid_result = grid.fit(X, Y, groups=groups)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    f = open("./reports/grid_search" + start_time + ".tx"
                                                    "t", "a+")
    for mean, stdev, param in zip(means, stds, params):
        f.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        print("%f (%f) with: %r" % (mean, stdev, param))
    f.close()
    return grid.best_estimator_


def train(X_train, y_train, X_test, y_test, left_out_participant="", conf_matrix=None):
    # check to see if we are compiling using just a single GPU
    if len(gpus) <= 1:
        print("[INFO] training with 1 GPU...")
        model = model_I((X_train.shape[1], X_train.shape[2], 1), n_classes=3)

    # otherwise, we are compiling using multiple GPUs
    else:
        print("[INFO] training with {} GPUs...".format(gpus))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = model_I((X_train.shape[1], X_train.shape[2], 1), n_classes=3)

        # make the model parallel
        model = multi_gpu_model(model, gpus=len(gpus))

    model.fit(X_train, y_train, epochs=config.get("cnn_params")['epochs'], validation_data=(X_test, y_test),
              batch_size=config.get("cnn_params")['batch_size'],
              # callbacks=[history, early_stopping])
              callbacks=[history])
    if len(gpus) > 1:
        prediction_classes = model.predict(X_test)
        prediction_classes = np.argmax(prediction_classes, axis=1)
    else:
        prediction_classes = model.predict_classes(X_test)
    c = [np.where(r == 1)[0][0] for r in y_test]
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


def grid_search_over_window_size():
    for wl in [1000, 2000, 3000, 4000, 5000]:
        print('testing w: ' + str(wl))
        X, Y, groups = get_grouped_windows_for_exerices(False, config, window_length=wl)
        (train, test) = split_train_test(X, Y, groups)
        model = model_I((train[0].shape[1], train[0].shape[2], 1))
        model.fit(train[0], train[1], epochs=config.get("cnn_params")['epochs'], validation_data=(test[0], test[1]),
                  batch_size=config.get("cnn_params")['batch_size'], )


def split_train_test(X, y, groups, n_classes=nb_classes, extra_features=None):
    gss = GroupShuffleSplit(test_size=0.2)
    train_indexes, test_indexes = gss.split(X, y, groups=groups).next()
    X_train = X[train_indexes]
    y_train = np_utils.to_categorical(y[train_indexes] - 1, n_classes)
    groups_train = groups[train_indexes]
    X_test = X[test_indexes]
    y_test = np_utils.to_categorical(y[test_indexes] - 1, n_classes)
    groups_test = groups[test_indexes]
    if extra_features is not None:
        X_extra_train = extra_features[train_indexes]
        X_extra_test = extra_features[test_indexes]
        return ((X_train, X_extra_train, y_train, groups_train), (X_test, X_extra_test, y_test, groups_test))

    return ((X_train, y_train, groups_train), (X_test, y_test, groups_test))


if __name__ == "__main__":
    X, Y, classes, groups = get_grouped_windows_for_rep_transistion(False, config, use_exercise_code_as_group=True)
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, extra_features=classes)
    for t in test[2]:
        s = ""
        if t[0] == 0:
            s += "0"
        else:
            s += "1"
    print(s)
    model = model_rep_transistions((tra[0].shape[1], tra[0].shape[2], 1), n_classes=2)
    model.fit([tra[0], tra[1]], tra[2], epochs=config.get("cnn_params")['epochs'],
              # validation_data=(test[0], test[1]),
              batch_size=config.get("cnn_params")['batch_size'],
              # callbacks=[history, early_stopping])
              )
    groups = np.unique(test[3])
    np.set_printoptions(linewidth=np.inf)
    for g in groups:
        indexes = np.argwhere(tra[0] == g)
        x = test[0][indexes]
        preds = model.predict(tra[0])
        print(test[2][indexes,0])
        print(preds[0][:,0])

    # X, Y, groups = get_grouped_windows_for_exerices(False, config)
    # (tra, test) = split_train_test(X, Y, groups)
    # model = model_I((tra[0].shape[1], tra[0].shape[2], 1))
    # grid_search(model_I, tra[0], tra[1], tra[2])
