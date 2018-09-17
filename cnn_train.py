# set the matplotlib backend so figures can be saved in the background
import argparse
import os

# leave this line
import psutil
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, LeaveOneGroupOut

from constants import RANDOMNESS_SEED, WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z, ANKLE_ACCEL_X, ANKLE_ACCEL_Y, \
    ANKLE_ACCEL_Z, WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z, ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z, ANKLE_ROT_Z, \
    ANKLE_ROT_Y, ANKLE_ROT_X, WRIST_ROT_Z, WRIST_ROT_Y, WRIST_ROT_X
from rep_counting import count_predicted_reps, count_real_reps
from results import LeaveOneOutResults, Result8020
from utils import *

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

from data_loading import get_grouped_windows_for_exerices, get_grouped_windows_for_rep_transistion_per_exercise
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


class TrainingParameters:

    def __init__(self, window_step_slide, window_length):
        self.window_step_slide = window_step_slide
        self.window_length = window_length

    def __str__(self):
        return 'wl:{}, step:{}'.format(self.window_length, self.window_step_slide)


class TrainingRepCountingParameters(TrainingParameters):

    def __init__(self, exercise, window_step_slide, window_length, conv_layers=3, strides=(3, 1), layers=5):
        TrainingParameters.__init__(self, window_step_slide, window_length)
        self.conv_layers = conv_layers
        self.exercise = exercise
        self.strides = strides
        self.layers = layers

    def __str__(self):
        return 'ex:{}, '.format(self.exercise) + TrainingParameters.__str__(self)


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


def rep_counting_model(input_shape,
                       strides,
                       layers,
                       filters=None,
                       dropout=None,
                       inner_dense_layer_neurons=250, n_classes=2):
    # sensor numpy_data
    if dropout is None:
        dropout = [0.5, 0.5, 0.5, 05, 0.5]
    if filters is None:
        filters = [100, 25, 75, 75, 25]
    conv_input = Input(shape=input_shape)
    input = conv_input
    for i in range(0, layers):
        conv_output = Convolution2D(filters=filters[i], kernel_size=(10, 18), strides=strides,
                                    input_shape=input_shape,
                                    border_mode='same',
                                    data_format="channels_last")(input)
        act = Activation('relu')(conv_output)
        after_dropout = Dropout(dropout[i])(act)
        input = after_dropout

    flattened = Flatten()(after_dropout)

    # Merge and add dense layer
    pre_output = Dense(inner_dense_layer_neurons)(flattened)
    output = (Dense(n_classes))(pre_output)
    output2 = Activation('softmax')(output)

    # Define model with two inputs
    model = Model(inputs=[conv_input], outputs=[output2])
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def recognition_model_with_statistical_features(input_shape,
                                                strides,
                                                layers,
                                                filters=None,
                                                dropout=None,
                                                inner_dense_layer_neurons=250, n_classes=2):
    # sensor numpy_data
    if dropout is None:
        dropout = [0.5, 0.5, 0.5, 05, 0.5]
    if filters is None:
        filters = [100, 25, 75, 75, 25]
    conv_input = Input(shape=input_shape)
    input = conv_input
    for i in range(0, layers):
        conv_output = Convolution2D(filters=filters[i], kernel_size=(15, 18), strides=strides,
                                    input_shape=input_shape,
                                    border_mode='same',
                                    data_format="channels_last")(input)
        act = Activation('relu')(conv_output)
        after_dropout = Dropout(dropout[i])(act)
        input = after_dropout

    flattened = Flatten()(after_dropout)

    stat_features_input = Input(shape=(150,))
    merge_layer = concatenate([flattened, stat_features_input])
    # Merge and add dense layer
    pre_output = Dense(inner_dense_layer_neurons)(merge_layer)
    output = (Dense(n_classes))(pre_output)
    output2 = Activation('softmax')(output)

    # Define model with two inputs
    model = Model(inputs=[conv_input, stat_features_input], outputs=[output2])
    sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def model_I(input_shape,
            conv_layer_1_filters=100, dropout_1=0.5,
            conv_layer_2_filters=25, dropout_2=0.5,
            conv_layer_3_filters=75, dropout_3=0.5,
            conv_layer_4_filters=75, dropout_4=0.5,
            conv_layer_5_filters=25, dropout_5=0.5,
            first_layer_kernle_size=(15, 18),
            first_layer_strides=(3, 1),
            inner_dense_layer_neurons=250,
            n_classes=nb_classes):
    K.clear_session()
    model = Sequential()
    model.add(
        Convolution2D(filters=conv_layer_1_filters, kernel_size=first_layer_kernle_size, strides=first_layer_strides,
                      input_shape=input_shape,
                      border_mode='same',
                      data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_1))
    model.add(Convolution2D(filters=conv_layer_2_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_2))
    model.add(Convolution2D(filters=conv_layer_3_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_3))
    model.add(Convolution2D(filters=conv_layer_4_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_4))
    model.add(Convolution2D(filters=conv_layer_5_filters, kernel_size=(15, 18), strides=(3, 1), input_shape=input_shape,
                            border_mode='same',
                            data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_5))
    model.add(Flatten())

    model.add(Dense(inner_dense_layer_neurons))
    model.add(Dense(n_classes))
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


def grid_search_over_window_size(save_model=False):
    for wl in [1000, 1500, 2000, 3000, 4000]:
        print('testing w: ' + str(wl))
        result = Result8020("window_size_w=" + str(wl))
        X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config, window_length=wl,
                                                        augumentation=False)
        (train, test) = split_train_test(X, Y, groups, n_classes=11)
        model = model_I((train[0].shape[1], train[0].shape[2], 1))
        history = model.fit(train[0], train[1],
                            epochs=config.get("cnn_params")['epochs'],
                            validation_data=(test[0], test[1]),
                            batch_size=config.get("cnn_params")['batch_size'], )
        result.set_result(history.history["val_acc"])
        if save_model:
            model.save("recognition_model_null_wl_" + str(wl) + ".h5")


def split_train_test(X, y, groups, n_classes=nb_classes, extra_features=None, test_size=0.1):
    gss = GroupShuffleSplit(test_size=test_size, random_state=RANDOMNESS_SEED)
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


def grid_search_single_rep_counting_model(model, X, Y, groups):
    # define params
    print("start Gridsearch")

    # params
    conv_layer_neurons = [50]
    # dropout_rate = [0.4, 0.6, 0.8]
    # inner_dense_layer_neurons = [100, 250, 500]

    conv_layer_1_filters = [100]
    dropout_1 = [0.5]
    conv_layer_2_filters = [25]
    dropout_2 = [0.5]
    conv_layer_3_filters = [75]
    dropout_3 = [0.5]
    conv_layer_4_filters = [75]
    dropout_4 = [0.5]
    conv_layer_5_filters = [25]
    dropout_5 = [0.7]
    input_shape = [(X.shape[1], X.shape[2], 1)]
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
                      epochs=[25],
                      batch_size=[30])

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


def rep_counting_training(training_parameters, with_reporting=False, augmentation=False, exercises=None):
    data_per_exercise = get_grouped_windows_for_rep_transistion_per_exercise(training_params=training_parameters,
                                                                             config=config,
                                                                             use_exercise_code_as_group=True,
                                                                             augmentation=augmentation,
                                                                             exercises=exercises)
    for ex in data_per_exercise.keys():
        print(training_parameters[ex])
        X, classes, Y, groups = data_per_exercise[ex]
        (tra, test) = split_train_test(X, Y, groups, n_classes=2, extra_features=classes)
        np.set_printoptions(linewidth=np.inf)

        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            model = rep_counting_model((tra[0].shape[1], tra[0].shape[2], 1), strides=training_parameters[ex].strides,
                                       layers=training_parameters[ex].layers)
        # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            with tf.device("/cpu:0"):
                # initialize the model
                model = rep_counting_model((tra[0].shape[1], tra[0].shape[2], 1),
                                           strides=training_parameters[ex].strides,
                                           layers=training_parameters[ex].layers)

            # make the model parallel
            model = multi_gpu_model(model, gpus=len(gpus))

        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = model.fit(tra[0], tra[2], epochs=config.get("cnn_params")['epochs'],
                            validation_data=(test[0], test[2]),
                            batch_size=config.get("cnn_params")['batch_size'],
                            # callbacks=[history, early_stopping])
                            )
        plot_learning_curves(history, file_name=str(ex))
        if augmentation:
            model.save("rep_counting_model_aug" + ex + ".h5")
        else:
            model.save("rep_counting_model_" + ex + ".h5")

        if with_reporting:
            f = open("./reports/single_exercise_rep_counting" + start_time + ".tx"
                                                                             "t", "a+")

            f.write((ex) + "\n")

            groups = np.unique(test[3])
            np.set_printoptions(linewidth=np.inf)
            errors = 0
            total_truth_reps = 0
            f.write("val acc " + str(history.history['val_acc'][-1]) + "\n")
            f.write("acc " + str(history.history['acc'][-1]) + "\n")
            for g in groups:
                f.write("ex id " + str(g) + "\n")
                indexes = np.argwhere(test[3] == g)
                x = test[0][indexes, :, :, :]
                x = np.squeeze(x)
                x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
                # x_extra = test[1][indexes, :]
                # x_extra = np.squeeze(x_extra, axis=2)
                preds = model.predict(x)
                truth = test[2][indexes, 0].astype(np.int32).ravel()
                preds = 1 - preds.argmax(axis=1)
                reps_truth = count_real_reps(truth)
                reps_pred = count_predicted_reps(preds)
                total_truth_reps += reps_truth
                abs_err = abs(reps_truth - reps_pred)
                errors += abs_err

                f.write("*****************\n")
                f.write(str(truth) + "\n")
                f.write(str(preds) + "\n")
                f.write("*****************\n")
                f.write("predicted " + str(reps_pred) + "\n")
                f.write("truth " + str(reps_truth) + "\n")
                f.write("difference " + str(reps_pred - reps_truth) + "\n")
            # f.write(truth)
            # f.write(preds)
            f.write('errors: ' + str(errors) + "\n")
            f.write('total reps: ' + str(total_truth_reps) + "\n")
            f.write("Relative total error " + str(float(errors / total_truth_reps)) + "\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
    if with_reporting:
        f.close()


def best_rep_counting_models():
    data_per_exercise = get_grouped_windows_for_rep_transistion_per_exercise(config, True)
    for ex in data_per_exercise.keys():
        X, classes, Y, groups = data_per_exercise[ex]
        Y = np_utils.to_categorical(Y - 1, 2)

        if len(gpus) <= 1:
            print("[INFO] training with 1 GPU...")
            model = rep_counting_model((X.shape[1], X.shape[2], 1), n_classes=2)
        # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(gpus))

            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            with tf.device("/cpu:0"):
                # initialize the model
                model = rep_counting_model((X.shape[1], X.shape[2], 1), n_classes=2)

            # make the model parallel
            model = multi_gpu_model(model, gpus=len(gpus))

        sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(X, Y, epochs=config.get("cnn_params")['epochs'],
                  batch_size=config.get("cnn_params")['batch_size'])
        model.save("rep_counting_model_" + ex + ".h5")


def rearrange_sensor_order(X):
    permutation = [WRIST_ACCEL_X, WRIST_ACCEL_Y, WRIST_ACCEL_Z, ANKLE_ACCEL_X, ANKLE_ACCEL_Y, ANKLE_ACCEL_Z,
                   WRIST_GYRO_X, WRIST_GYRO_Y, WRIST_GYRO_Z, ANKLE_GYRO_X, ANKLE_GYRO_Y, ANKLE_GYRO_Z,
                   WRIST_ROT_X, WRIST_ROT_Y, WRIST_ROT_Z, ANKLE_ROT_X, ANKLE_ROT_Y, ANKLE_ROT_Z]
    i = np.argsort(permutation)
    return X[:, :, i, :]


def best_recognition_model(kernel_size=(15, 3),
                           strides=(1, 3), with_null_class=True):
    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=with_null_class)
    print(X.shape)
    print(X.mean(axis=1).shape)
    if with_null_class:
        number_of_classes = 11
    else:
        number_of_classes = 10
    (tra, test) = split_train_test(X, Y, groups, n_classes=number_of_classes, test_size=0.2)
    Y = np_utils.to_categorical(Y - 1, number_of_classes)
    if with_null_class:
        nb_classes = 11
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1), first_layer_kernle_size=kernel_size,
                    first_layer_strides=strides,
                    n_classes=number_of_classes)
    model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
              batch_size=config.get("cnn_params")['batch_size'],
              validation_data=(test[0], test[1])
              # callbacks=[history, early_stopping])
              # callbacks=[history]
              )
    model.save("recognition_model_with_null.h5")


def grid_search_over_convolution_parameters():
    best_recognition_model(kernel_size=(15, 3), strides=(1, 3))
    # best_recognition_model(kernel_size=(15, 9), strides=(1, 6))
    best_recognition_model(kernel_size=(15, 9), strides=(1, 9))
    best_recognition_model(kernel_size=(15, 18), strides=(1, 1))
    # best_recognition_model(kernel_size=(15, 18), strides=(3, 1))
    # best_recognition_model(kernel_size=(20, 18), strides=(3, 1))


def init_best_rep_counting_models_params():
    ex_to_rep_traning_model_params = {}
    ex_to_rep_traning_model_params["Crunches"] = TrainingRepCountingParameters(exercise="Crunches", window_length=200,
                                                                               window_step_slide=0.10)
    ex_to_rep_traning_model_params["KB Press"] = TrainingRepCountingParameters(exercise="Kb Press", window_length=150,
                                                                               window_step_slide=0.10)
    ex_to_rep_traning_model_params["Wall balls"] = TrainingRepCountingParameters(exercise="Wall balls",
                                                                                 window_length=200,
                                                                                 window_step_slide=0.10)
    ex_to_rep_traning_model_params["Push ups"] = TrainingRepCountingParameters("Push ups", window_length=120,
                                                                               window_step_slide=0.05, layers=5,
                                                                               strides=(1, 1))
    ex_to_rep_traning_model_params["Dead lifts"] = TrainingRepCountingParameters("Dead lifts", window_length=200,
                                                                                 window_step_slide=0.10, layers=3,
                                                                                 strides=(1, 1))
    ex_to_rep_traning_model_params["Burpees"] = TrainingRepCountingParameters("Burpees", window_length=250,
                                                                              window_step_slide=0.10)
    ex_to_rep_traning_model_params["Squats"] = TrainingRepCountingParameters("Squats", window_length=150,
                                                                             window_step_slide=0.10)
    ex_to_rep_traning_model_params["KB Squat press"] = TrainingRepCountingParameters("Kb Squat press",
                                                                                     window_length=150,
                                                                                     window_step_slide=0.05)
    ex_to_rep_traning_model_params["Box jumps"] = TrainingRepCountingParameters("Box Jumps", window_length=200,
                                                                                window_step_slide=0.10)
    ex_to_rep_traning_model_params["Pull ups"] = TrainingRepCountingParameters("Pull ups", window_length=200,
                                                                               window_step_slide=0.10)
    return ex_to_rep_traning_model_params


def exercise_vs_null_training():
    result = Result8020("exercise_vs_null_training")
    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=True)
    Y[Y != 11] = 1
    Y[Y == 11] = 0
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=2)
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1]),
                        callbacks=[early_stopping])
    # callbacks=[history])
    result.set_result(history.history["val_acc"])


def hand_training():
    result = Result8020("hand_training")
    X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config, augumentation=False,
                                                    with_null_class=False,
                                                    )
    print("first")
    print(X.shape)
    X = X[:, :, WRIST_ACCEL_X:WRIST_ROT_Z + 1, :]
    (tra, test) = split_train_test(X, Y, groups, n_classes=nb_classes, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=nb_classes,
                    first_layer_kernle_size=(15, 9))
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1])
                        # callbacks=[history, early_stopping])
                        # callbacks=[history]
                        )
    result.set_result(history.history['val_acc'])


def foot_training():
    result = Result8020("foot_training")
    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=False)
    X = X[:, :, ANKLE_ACCEL_X:, :]
    (tra, test) = split_train_test(X, Y, groups, n_classes=nb_classes, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    first_layer_kernle_size=(15, 9),
                    n_classes=nb_classes)
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1])
                        # callbacks=[history, early_stopping])
                        # callbacks=[history]
                        )
    result.set_result(history.history['val_acc'])


def acc_hand_training():
    result = Result8020("acc_hand_training")

    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=True)
    X = X[:, :, WRIST_ACCEL_X:WRIST_ACCEL_Z + 1, :]
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=2)
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1])
                        # callbacks=[history, early_stopping])
                        # callbacks=[history]
                        )
    result.set_result(history.history['val_acc'])


def gyro_hand_training():
    result = Result8020("gyro_hand_training")
    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=True)
    X = X[:, :, WRIST_GYRO_X:WRIST_GYRO_Z + 1, :]
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=2)
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1])
                        # callbacks=[history, early_stopping])
                        # callbacks=[history]
                        )
    result.set_result(history.history['val_acc'])


def rot_hand_training():
    result = Result8020("rot_hand_training")
    X, Y, groups = get_grouped_windows_for_exerices(False, config, augumentation=False, with_null_class=True)
    X = X[:, :, WRIST_ROT_X:WRIST_ROT_Z + 1, :]
    (tra, test) = split_train_test(X, Y, groups, n_classes=2, test_size=0.2)
    model = model_I((tra[0].shape[1], tra[0].shape[2], 1),
                    n_classes=2)
    history = model.fit(tra[0], tra[1], epochs=config.get("cnn_params")['epochs'],
                        batch_size=config.get("cnn_params")['batch_size'],
                        validation_data=(test[0], test[1])
                        # callbacks=[history, early_stopping])
                        # callbacks=[history]
                        )
    result.set_result(history.history['val_acc'])


def center_data(X):
    return X - X.mean(axis=1, keepdims=True)


def train_model_with_statistical_features():
    X, X_stat_features, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=True, config=config,
                                                                     augumentation=False,
                                                                     with_null_class=False)

    X = center_data(X)
    (tra, test) = split_train_test(X, Y, groups, n_classes=10, test_size=0.2, extra_features=X_stat_features)
    print(tra[1].shape)
    model = recognition_model_with_statistical_features((tra[0].shape[1], tra[0].shape[2], 1),
                                                        n_classes=10, strides=(3, 1), layers=5
                                                        )
    model.fit([tra[0], tra[1]], tra[2], epochs=config.get("cnn_params")['epochs'],
              batch_size=config.get("cnn_params")['batch_size'],
              validation_data=([test[0], test[1]], test[2])
              # callbacks=[history, early_stopping])
              # callbacks=[history]
              )


def cross_validated_recognition_results():
    results = LeaveOneOutResults("leave_one_out_recognition_results")

    X, Y, groups = get_grouped_windows_for_exerices(with_feature_extraction=False, config=config,
                                                    augumentation=False,
                                                    with_null_class=False)
    logo = LeaveOneGroupOut()
    gss = GroupShuffleSplit(test_size=0.2, random_state=RANDOMNESS_SEED)
    # logo.get_n_splits(X, Y, groups)
    #
    # logo.get_n_splits(groups=groups)  # 'groups' is always required
    for train_index, test_index in logo.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np_utils.to_categorical(Y[train_index] - 1, 10), np_utils.to_categorical(Y[test_index] - 1,
                                                                                                   10)

        model = model_I((X_train.shape[1], X_train.shape[2], 1), first_layer_kernle_size=(15, 3),
                        first_layer_strides=(1, 3),
                        n_classes=10)
        history = model.fit(X_train, y_train,
                            epochs=config.get("cnn_params")['epochs'],
                            batch_size=config.get("cnn_params")['batch_size'],
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping])
        predicted_values = model.predict_classes(X_test)
        truth_values = argmax(y_test, axis=1)
        results.set_results(truth_values, predicted_values, history.history['val_acc'])


if __name__ == "__main__":
    cross_validated_recognition_results()
