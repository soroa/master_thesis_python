# set the matplotlib backend so figures can be saved in the background
from keras import Sequential
from keras.layers import Convolution1D, Activation, Flatten, Dropout, Dense, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils

from data_loading import get_windowed_exerices_feautres_for_training_data
from tensorflow.python.client import device_lib


def ski_jumps_model():
    #
    model = Sequential()
    model.add(
        Convolution1D(nb_filter=32, kernel_size=10, input_shape=(X_train.shape[1], X_train.shape[2]), padding="same",
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

print(device_lib.list_local_devices())
X_train, y_train, X_test, y_test = get_windowed_exerices_feautres_for_training_data(False)


# X_train, y_train = remove_nans_raw(X_train, y_train)
# X_test, y_test = remove_nans_raw(X_test, y_test)

nb_classes= 10



#
model = Sequential()
model.add(
    Convolution1D(nb_filter=32, kernel_size=10, input_shape=(X_train.shape[1], X_train.shape[2]), padding="same",
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


y_train = np_utils.to_categorical(y_train-1, nb_classes)
y_valid = np_utils.to_categorical(y_test-1, nb_classes)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 15
model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_test, y_valid), batch_size=16)