import os
import random
import re

import numpy as np

from constants import *
from data_features_extraction import extract_features_for_single_reading, extract_features


def remove_nans_raw(X, Y):
    mask = ~np.isnan(X).any(axis=2)
    X = X[mask]
    Y = Y[mask[:, 0]]
    return [X, Y]


def get_total_number_of_exercises():
    ex_names = os.listdir(numpy_exercises_data_path)
    count = 0
    for ex in ex_names:
        count += len(os.listdir(numpy_exercises_data_path + '/' + ex))
    return count


def get_windowed_exerices_feautres_for_training_data(with_feature_extraction):
    ex_folders = os.listdir(numpy_exercises_data_path)
    exercises_count_total = get_total_number_of_exercises()
    train_indexs = list(range(0, exercises_count_total))
    while len(train_indexs) > int(0.75 * exercises_count_total):
        index = random.randrange(len(train_indexs))
        del train_indexs[index]

    windows_train = []
    windows_test = []
    window_length = 3000
    labels_train = []
    labels_test = []
    index = 0
    for ex_folder in ex_folders:
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex_folder]
        for exercise_file in exericse_readings_list:
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            windows_for_exercise = extract_windows(exercise, window_length)
            if index in train_indexs:
                windows_train += windows_for_exercise
            else:
                windows_test += windows_for_exercise
            for i in range(0, len(windows_for_exercise)):
                if index in train_indexs:
                    labels_train.append(label)
                else:
                    labels_test.append(label)
            index += 1
    X_train = np.asarray(windows_train)
    Y_train = np.asarray(labels_train)
    X_test = np.asarray(windows_test)
    Y_test = np.asarray(labels_test)
    if with_feature_extraction:
        X_train_features = extract_features(X_train[:, 1:, :])
        X_test_features = extract_features(X_test[:, 1:, :])
        return [X_train_features, Y_train, X_test_features, Y_test]
    else:
        return [np.transpose(X_train, (0, 2, 1))[:, :, 1:], Y_train, np.transpose(X_test, (0, 2, 1))[:, :, 1:], Y_test]


def extract_windows(exercise_reading, window_length_in_ms):
    windows = []
    for i in range(0, exercise_reading.shape[1] - int(window_length_in_ms / 10), int(window_length_in_ms / 10 * 0.20)):
        windows.append(exercise_reading[:, i:i + int(window_length_in_ms / 10)])
    return windows


# get_windowed_exerices_raw_training_data()


def get_reps_data_features():
    train_ids = get_exercise_ids()
    initial_length = len(train_ids)
    while len(train_ids) > int(0.95 * initial_length):
        index = random.randrange(len(train_ids))
        del train_ids[index]

    reps_test = np.zeros((0, 150))
    labels_test = np.zeros((0, 1))
    reps_train = np.zeros((0, 150))
    labels_train = np.zeros((0, 1))
    reps_ex_names = os.listdir(numpy_reps_data_path)
    for ex in reps_ex_names:
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex]
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps:
            rep = np.load(numpy_reps_data_path + "/" + ex + '/' + r_name)
            rep_features = extract_features_for_single_reading(rep[1:, :]).reshape((1, 150))
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", r_name)
            rep_ex_id = rep_ex_id_plus_rep_num[0:len(rep_ex_id_plus_rep_num) - 1]
            if rep_ex_id in train_ids:
                reps_train = np.append(reps_train, rep_features, axis=0)
                labels_train = np.append(labels_train, label)
            else:
                reps_test = np.append(reps_test, rep_features, axis=0)
                labels_test = np.append(labels_test, label)

    return [reps_train, labels_train, reps_test, labels_test]


def get_reps_labels():
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


def get_exercise_ids():
    ex_folders = os.listdir(numpy_exercises_data_path)
    ids = []
    for ex_folder in ex_folders:
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in exericse_readings_list:
            ex_id = re.sub("[^0-9]", "", exercise_file)
            ids.append(ex_id)
    return ids


def get_total_number_of_reps():
    reps_ex_names = os.listdir(numpy_reps_data_path)
    count = 0
    for ex in reps_ex_names:
        count += len(os.listdir(numpy_reps_data_path + '/' + ex))
    print(count)
    return count


def add_0_padding_to_rep(rep, final_length):
    padded_rep = np.zeros([rep.shape[0], final_length])
    padded_rep[:, 0:rep.shape[1]] = rep
    return padded_rep
