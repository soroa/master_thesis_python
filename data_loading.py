import os
import random
import re

from scipy.stats import mode

from utils import *

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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


def get_group_for_id(id, config):
    part_to_ex = config.get("participant_to_ex_code_map")
    i = 0
    for name in list(part_to_ex.keys()):
        if int(id) in part_to_ex[name]:
            return i
        i += 1
    return None


def get_experience_level_for_ex_id(id, config):
    part_to_ex = config.get("participant_to_ex_code_map")
    for name in list(part_to_ex.keys()):
        if int(id) in part_to_ex[name]:
            return EXPERIENCE_LEVEL_MAP[name]
    return None


def get_grouped_windows_for_exerices(with_feature_extraction,
                                     config=None, window_length=None):
    ex_folders = os.listdir(numpy_exercises_data_path)
    if (window_length is None):
        window_length = config.get("data_params")["window_length"]
    groups = []
    windows = []
    labels = []
    for ex_folder in ex_folders:
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex_folder]
        for exercise_file in exericse_readings_list:
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            windows_for_exercise = extract_windows(exercise, window_length)
            windows += windows_for_exercise
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))
            current_group = get_group_for_id(exercise_code, config)
            for i in range(0, len(windows_for_exercise)):
                labels.append(label)
                groups.append(current_group)

    X = np.asarray(windows)
    X = X[:, 1:, :]
    Y = np.asarray(labels)
    groups = np.asarray(groups)
    if config is not None:
        sensors = config.get("sensors")
        sensor_mask = np.ones(18).astype(np.bool)
        if "acc" not in sensors:
            sensor_mask[0:3] = False
            sensor_mask[9:12] = False
        if "gyro" not in sensors:
            sensor_mask[3:6] = False
            sensor_mask[12:15] = False
        if "rot" not in sensors:
            sensor_mask[6:9] = False
            sensor_mask[15:] = False
        positions = config.get("sensor_positions")
        if "wrist" not in positions:
            sensor_mask[0:9] = False
        elif "foot" not in positions:
            sensor_mask[9:] = False

        X = X[:, sensor_mask, :]

    if with_feature_extraction:
        X = extract_features(X)
        X_features = np.nan_to_num(X)
        return [X_features, Y, groups]
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.nan_to_num(X)
        return [np.transpose(X, (0, 2, 1, 3)), Y, groups]


def get_rep_count_for_exericse_id(ex_id):
    reps_ex_names = os.listdir(numpy_reps_data_path)
    reps = 0
    for ex in reps_ex_names:
        single_reps = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps:
            if ex_id in r_name:
                reps += 1
    return reps


def get_exercise_readings():
    ex_folders = os.listdir(numpy_exercises_data_path)
    exercises = []
    labels = []
    rep_counts = []
    a = True
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue;
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex_folder]
        for exercise_file in exericse_readings_list:
            if a:
                print(exercise_file)
                a = False
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            exercises.append(exercise[1:, :])
            exercise_id = (re.sub("[^0-9]", "", exercise_file))
            rep_count = get_rep_count_for_exericse_id(exercise_id)
            labels.append(label)
            rep_counts.append(rep_count)
    Y = np.asarray(labels)
    return exercises, Y, rep_counts


def extract_windows(exercise_reading, window_length_in_ms):
    windows = []
    for i in range(0, exercise_reading.shape[1] - int(window_length_in_ms / 10), int(window_length_in_ms / 10 * 0.20)):
        windows.append(exercise_reading[:, i:i + int(window_length_in_ms / 10)])
    return windows


def get_reps_with_features():
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


def get_reps_raw_with_zero_padding():
    train_ids = get_exercise_ids()
    initial_length = len(train_ids)
    while len(train_ids) > int(0.90 * initial_length):
        index = random.randrange(len(train_ids))
        del train_ids[index]
    reps_test = np.zeros((0, 18, 500))
    labels_test = np.zeros((0, 1))
    reps_train = np.zeros((0, 18, 500))
    labels_train = np.zeros((0, 1))
    reps_ex_names = os.listdir(numpy_reps_data_path)
    for ex in reps_ex_names:
        label = EXERCISE_NAME_TO_CLASS_LABEL[ex]
        single_reps_file_names = os.listdir(numpy_reps_data_path + '/' + ex)
        for r_name in single_reps_file_names:
            rep = np.load(numpy_reps_data_path + "/" + ex + '/' + r_name)[1:]
            rep_padded = np.zeros((1, 18, 500))
            rep_padded[0, :, 0:rep.shape[1]] = rep
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", r_name)
            rep_ex_id = rep_ex_id_plus_rep_num[0:len(rep_ex_id_plus_rep_num) - 1]
            if rep_ex_id in train_ids:
                reps_train = np.append(reps_train, rep_padded, axis=0)
                labels_train = np.append(labels_train, label)
            else:
                reps_test = np.append(reps_test, rep_padded, axis=0)
                labels_test = np.append(labels_test, label)

    return [reps_train.transpose((0, 2, 1)), labels_train, reps_test.transpose((0, 2, 1)), labels_test]


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


def plot_exercise_per_id(exercise_id, person=None):
    ex_folders = os.listdir(numpy_exercises_data_path)
    readings = None
    for ex_folder in ex_folders:
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in exericse_readings_list:
            if exercise_id in exercise_file:
                readings = np.load(numpy_exercises_data_path + "/" + ex_folder + "/" + exercise_file)
                break
    if readings is None:
        print("Readings not found")
        return
    time = range(0, readings.shape[1])
    plt.figure()
    if person is not None:
        plt.suptitle(person)
    for i in range(1, 10):
        plt.subplot(9, 1, i)
        if (i < 4):
            c = 'b'
        elif (i < 7):
            c = 'g'
        else:
            c = 'r'
        plt.plot(time, readings[i, :], c)

    plt.figure()
    if person is not None:
        plt.suptitle(person + " foot")
    for i in range(1, 10):
        plt.subplot(9, 1, i)
        if (i < 4):
            c = 'b'
        elif (i < 7):
            c = 'g'
        else:
            c = 'r'
        plt.plot(time, readings[i + 9, :], c)


def get_reps_duration_map():
    reps_folders = os.listdir(numpy_reps_data_path)
    ex_code_to_rep_count_map = {}
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        for rep_readings_file_name in reps_readings_list:
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", rep_readings_file_name)
            rep_ex_id = int(rep_ex_id_plus_rep_num[0:3])
            if rep_ex_id not in ex_code_to_rep_count_map.keys():
                rep = np.load(numpy_reps_data_path + rep_folder + "/" + rep_readings_file_name)
                length = rep.shape[1]
                if length < 100:
                    continue
                # print(length)
                len_rounded = int(50 * round(float(length) / 50))
                ex_code_to_rep_count_map[rep_ex_id] = len_rounded
                # print(len_rounded)
                # print()

    return ex_code_to_rep_count_map


def does_window_contain_rep_transition(window_endtime, rep_duration, transition_duration, window_length):
    if window_endtime % rep_duration <= window_length - transition_duration / 2 and window_endtime % rep_duration >= transition_duration / 2:
        return True
    return False


def get_exercise_labeled_transition_windows(exercise_reading, window_length_in_ms, exercise_code, transition_duration,
                                            ex_code_class=None, with_ex_code_as_feature=False):
    reps_duration_map = get_reps_duration_map()
    rep_duration = reps_duration_map[exercise_code]
    windows = []
    labels = []
    for start in range(0, exercise_reading.shape[1] - int(window_length_in_ms / 10),
                       int(window_length_in_ms / 10 * 0.20)):
        stop = start + int(window_length_in_ms / 10)
        if with_ex_code_as_feature:
            with_ex_code_feature = np.append(exercise_reading[:, start:stop], np.full((1, 100), ex_code_class), axis=0)
            windows.append(with_ex_code_feature)
        else:
            windows.append(exercise_reading[:, start:stop])
        if does_window_contain_rep_transition(stop, rep_duration, transition_duration, window_length_in_ms / 10):
            labels.append(True)
        else:
            labels.append(False)
    return (windows, labels)


def calculate_longest_and_shortest_rep_per_exercise():
    reps_folders = os.listdir(numpy_reps_data_path)
    ex_code_to_rep_durations = [[], [], [], [], [], [], [], [], [], []]
    ex_code_seen = []
    for rep_folder in reps_folders:
        reps_readings_list = os.listdir(numpy_reps_data_path + '/' + rep_folder)
        for rep_readings_file_name in reps_readings_list:
            rep_ex_id_plus_rep_num = re.sub("[^0-9]", "", rep_readings_file_name)
            rep_ex_id = int(rep_ex_id_plus_rep_num[0:3])
            rep = np.load(numpy_reps_data_path + rep_folder + "/" + rep_readings_file_name)
            length = rep.shape[1]
            len_rounded = int(50 * round(float(length) / 50))
            if rep_ex_id in ex_code_seen or len_rounded < 150:
                continue
            else:
                ex_code_seen += [rep_ex_id]
            ex_code_to_rep_durations[EXERCISE_NAME_TO_CLASS_LABEL[rep_folder] - 1] += [len_rounded]
            # print(len_rounded)
            # print()
    for i in range(0, len(ex_code_to_rep_durations)):
        ex_code_to_rep_durations[i] = (
            min(ex_code_to_rep_durations[i]), mode(ex_code_to_rep_durations[i])[0][0], max(ex_code_to_rep_durations[i]))
    return ex_code_to_rep_durations


def get_grouped_windows_for_rep_transistion(with_feature_extraction,
                                            config=None, use_exercise_code_as_group = False):
    ex_folders = os.listdir(numpy_exercises_data_path)
    window_length = config.get("data_params")["window_length"]
    rep_transition_duration = 50
    window_length_in_ms = 1000
    groups = []
    windows = []
    transition_labels = []
    classes = []
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in exericse_readings_list:
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))

            (windows_for_exercise, transition_labels_for_ex) = get_exercise_labeled_transition_windows(exercise,
                                                                                                      window_length_in_ms,
                                                                                                      exercise_code,
                                                                                                      rep_transition_duration,
                                                                                                      EXERCISE_NAME_TO_CLASS_LABEL[
                                                                                                          ex_folder],
                                                                                                      with_ex_code_as_feature=False)
            windows += windows_for_exercise
            transition_labels += transition_labels_for_ex
            if use_exercise_code_as_group:
                current_group = exercise_code
            else:
                current_group = get_group_for_id(exercise_code, config)
            for i in range(0, len(windows_for_exercise)):
                groups.append(current_group)
                classes.append(EXERCISE_NAME_TO_CLASS_LABEL[ex_folder])

    X = np.asarray(windows)
    X = X[:, 1:, :]
    Y = np.asarray(transition_labels)
    classes = np.asarray(classes).reshape((len(classes), 1))
    groups = np.asarray(groups)
    # if config is not None:
    #     sensors = config.get("sensors")
    #     sensor_mask = np.ones(18).astype(np.bool)
    #     if "acc" not in sensors:
    #         sensor_mask[0:3] = False
    #         sensor_mask[9:12] = False
    #     if "gyro" not in sensors:
    #         sensor_mask[3:6] = False
    #         sensor_mask[12:15] = False
    #     if "rot" not in sensors:
    #         sensor_mask[6:9] = False
    #         sensor_mask[15:] = False
    #     positions = config.get("sensor_positions")
    #     if "wrist" not in positions:
    #         sensor_mask[0:9] = False
    #     elif "foot" not in positions:
    #         sensor_mask[9:] = False
    #
    #     X = X[:, sensor_mask, :]

    if with_feature_extraction:
        X = extract_features(X)
        X_features = np.nan_to_num(X)
        return [X_features, Y, groups, classes]
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = np.nan_to_num(X)
        return [np.transpose(X, (0, 2, 1, 3)), Y,classes, groups]


def get_experience_level_data(config):
    ex_folders = os.listdir(numpy_exercises_data_path)
    window_length = config.get("data_params")["window_length"]
    groups = []
    windows = []
    experience_labels = []
    for ex_folder in ex_folders:
        if not os.path.isdir(numpy_exercises_data_path + ex_folder):
            continue;
        exericse_readings_list = os.listdir(numpy_exercises_data_path + '/' + ex_folder)
        for exercise_file in exericse_readings_list:
            exercise = np.load(numpy_exercises_data_path + "/" + ex_folder + '/' + exercise_file)
            windows_for_exercise = extract_windows(exercise, window_length)
            windows += windows_for_exercise
            exercise_code = int(re.sub("[^0-9]", "", exercise_file))
            current_group = get_group_for_id(exercise_code, config)
            experience = get_experience_level_for_ex_id(exercise_code, config)
            for i in range(0, len(windows_for_exercise)):
                experience_labels.append(experience)
                groups.append(current_group)

    X = np.asarray(windows)
    X = X[:, 1:, :]
    Y = np.asarray(experience_labels)
    groups = np.asarray(groups)
    if config is not None:
        sensors = config.get("sensors")
        sensor_mask = np.ones(18).astype(np.bool)
        if "acc" not in sensors:
            sensor_mask[0:3] = False
            sensor_mask[9:12] = False
        if "gyro" not in sensors:
            sensor_mask[3:6] = False
            sensor_mask[12:15] = False
        if "rot" not in sensors:
            sensor_mask[6:9] = False
            sensor_mask[15:] = False
        positions = config.get("sensor_positions")
        if "wrist" not in positions:
            sensor_mask[0:9] = False
        elif "foot" not in positions:
            sensor_mask[9:] = False

        X = X[:, sensor_mask, :]

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = np.nan_to_num(X)
    return [np.transpose(X, (0, 2, 1, 3)), Y, groups]
