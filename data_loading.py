import os

import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from constants import *


# def get_entire_exerices_training_data():
# X = np.zeros((get_total_number_of_exercises(), 19, 5000))
# ex_folders = os.listdir(numpy_reps_data_path)
# index = 0
# for ex_folder in ex_folders:
#     exericse_readings_list = os.listdir(numpy_reps_data_path + '/' + ex_folder)
#     for exercise_file in exericse_readings_list:
#         exercise = np.load(numpy_reps_data_path + "/" + ex_folder + '/' + exercise_file)
#         rep_padded = add_0_padding_to_rep(exercise, 5000)
#         X[index] = rep_padded
#         index += 1
# Y = get_labels()
# return {"X": X, "Y": Y}
from data_features_extraction import extract_features_for_single_reading


def get_reps_data():
    X = np.zeros((get_total_number_of_reps(), 19, 500))
    reps_ex_names = os.listdir(numpy_reps_data_path)
    index = 0
    for ex in reps_ex_names:
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps:
            rep = np.load(numpy_reps_data_path + "/" + ex + '/' + r_name)
            rep_padded = add_0_padding_to_rep(rep, 500)
            X[index] = rep_padded
            index += 1
    X = X[:, 1:X.shape[1], :]
    Y = get_labels()
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    encoded_Y = np_utils.to_categorical(encoded_Y)
    # 0 samples, timestamps, sensor readings
    return [np.transpose(X, (0, 2, 1)), encoded_Y]

def get_reps_features_data():
    X = np.zeros((get_total_number_of_reps(), 133))
    reps_ex_names = os.listdir(numpy_reps_data_path)
    index = 0
    for ex in reps_ex_names:
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps:
            rep = np.load(numpy_reps_data_path + "/" + ex + '/' + r_name)
            rep_features = extract_features_for_single_reading(rep[1:, :])
            X[index] = rep_features
            index += 1
    Y = get_labels()
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    encoded_Y = np_utils.to_categorical(encoded_Y)
    # 0 samples, timestamps, sensor readings
    return [X, encoded_Y]


def get_labels():
    Y = np.zeros((get_total_number_of_reps(), 1))
    reps_ex_names = os.listdir(numpy_reps_data_path)
    index = 0
    for ex in reps_ex_names:
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        current_label = EXERCISE_NAME_TO_CLASS_LABEL[ex]
        for r_name in single_reps:
            Y[index] = current_label
            index += 1
    return Y


def get_total_number_of_reps():
    reps_ex_names = os.listdir(numpy_reps_data_path)
    count = 0
    for ex in reps_ex_names:
        count += len(os.listdir(numpy_reps_data_path + '/' + ex))
    print(count)
    return count


def get_total_number_of_exercises():
    ex_names = os.listdir(numpy_exercises_data_path)
    count = 0
    for ex in ex_names:
        count += len(os.listdir(numpy_exercises_data_path + '/' + ex))
    print(count)
    return count


def add_0_padding_to_rep(rep, final_length):
    padded_rep = np.zeros([rep.shape[0], final_length])
    padded_rep[:, 0:rep.shape[1]] = rep
    return padded_rep

# get_total_number_of_exercises()
