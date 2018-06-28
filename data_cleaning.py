import collections
import os
import shutil
import sqlite3

import json
import matplotlib.pyplot as plt

from constants import *
from constants import copy_from_path, path, numpy_reps_data_path, numpy_exercises_data_path
from db_functions import *
from plot import addRepSeparators, interpolate

tables = [WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME]
reversed_ankle_watch_participant_ids = [42, 50, 51, 52, 53, 54, 57, 60, 62, 64, 65, 67, 68, 69]


def split_wrist_ankle_and_merge():
    for file in os.listdir(path):
        os.remove(path + file)

    for file in os.listdir(copy_from_path):
        shutil.copyfile(copy_from_path + file, path + file)

    dbs_names_before_merge = os.listdir(path)

    ankle_readings = []
    wrist_readings = []
    for name in dbs_names_before_merge:
        if "ankle" in name:
            ankle_readings.append(name)
        elif "wrist" in name:
            wrist_readings.append(name)
    print(wrist_readings)
    print(ankle_readings)
    merge_databases(wrist_readings, "wrist")
    merge_databases(ankle_readings, "ankle")
    for name in os.listdir(path):
        if "merged" in name:
            continue
        os.remove(path + name)


def merge_databases(names, position):
    first_name = names[0]
    first_db = sqlite3.connect(path + first_name)
    first_cursor = first_db.cursor()
    names.remove(first_name)
    for name in names:
        print(name)
        db = sqlite3.connect(path + name)
        current_participant_cursor = db.cursor()
        for table in tables:
            print(table)
            current_participant_cursor.execute('SELECT * FROM {tn}'.format(tn=table))
            readings = current_participant_cursor.fetchall()  # Returns the results as a list.
            for r in readings:
                if table is READINGS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?,?,?"
                elif table is EXERCISES_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                elif table is WORKOUTS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                first_cursor.execute('INSERT INTO {tn} VALUES({values_placeholder})'.format(tn=table,
                                                                                            values_placeholder=values_placeholder),
                                     r)
        current_participant_cursor.close()
    first_db.commit()
    first_cursor.close()
    os.rename(path + first_name, path + "merged_" + position)


def adjust_reversed_watch_position():
    print("****** AJUST UPSIDE DOWN WATCH READINGS*******")
    print("Running... ")
    db = sqlite3.connect(path + "merged_ankle")
    c = db.cursor()
    for id in reversed_ankle_watch_participant_ids:
        ex_ids = get_exercises_ids_for_workout_id(id, db)
        for ex_id in ex_ids:
            readings = np.array(c.execute(
                "SELECT * FROM {tn} WHERE exercise_id={exid}".format(tn=READINGS_TABLE_NAME,
                                                                     exid=ex_id[0])).fetchall())
            for entry in readings:
                values_as_string = entry[READING_VALUES]
                # print("before: " + values_as_string)
                vals_split = np.array(values_as_string.split(" "))[0:3]
                vals_as_float = vals_split.astype(np.float)
                adjusted_values = np.multiply(vals_as_float, np.array([-1, -1, 1]))
                adjust_values_as_string = str(adjusted_values[0]) + " " + str(adjusted_values[1]) + " " + str(
                    adjusted_values[2])
                # print("after: " + adjust_values_as_string)
                entry[READING_VALUES] = adjust_values_as_string
                c.execute(
                    "UPDATE {tn} SET [values]='{adjusted_values}' WHERE id={id} ".format(tn=READINGS_TABLE_NAME,
                                                                                         id=entry[READING_ID],
                                                                                         adjusted_values=adjust_values_as_string))
    db.commit()
    c.close()


# def print_participant_name():
#     for name in dbs_names:
#         db = sqlite3.connect(path + name)
#         b_cursor = db.cursor()
#         b_cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
#         participants = np.array(b_cursor.fetchall())
#         for p in participants:
#             print(name + " " + p[0])


def remove_zero_reps(position):
    print("\n")
    print("***** Removing 0 reps")
    print("\n")
    db = sqlite3.connect(path + "merged_" + position)
    cursor = db.cursor()
    cursor.execute('DELETE FROM {tn} WHERE rep_count=0'.format(tn=READINGS_TABLE_NAME))
    db.commit()
    cursor.close()


def check_exercises_for_participant(db, participant, position):
    ex_codes = get_exercise_codes_for_participants(db, participant)
    for code in WORKOUT:
        if code not in ex_codes:
            print(participant + " did not perform " + EXERCISE_CODES_TO_NAME[code] + " on " + position)
    check_readings_for_participant(db, participant, position)
    # check that readings are there


def check_readings_for_participant(db, participant, position):
    ex_ids = get_exercises_ids_for_participants(db, participant)
    for id in ex_ids:
        readings = get_readings_for_exercise(db, id)
        if readings.size == 0:
            print(participant + " " + position + "  readings are missing for exercise " + get_exercise_name_for_id(db,
                                                                                                                   id))
        else:
            check_all_sensors_were_recorded(readings, participant, position, id[0])


def check_all_sensors_were_recorded(readings, participant, position, ex_id):
    sensor_codes = readings[:, READING_SENSOR_TYPE]
    occurences = collections.Counter(sensor_codes)
    if occurences[str(ACCELEROMETER_CODE)] == 0:
        print(participant + " " + position + " acc data is missing for ex_id " + str(ex_id))
    if occurences[str(GYROSCOPE_CODE)] == 0:
        print(participant + " " + position + " gyro data is missing for ex_id " + str(ex_id))
    if occurences[str(ROTATION_MOTION)] == 0:
        print(participant + " " + position + " rot data is missing for ex_id " + str(ex_id))


def were_all_sensors_recorded(readings):
    sensor_codes = readings[:, READING_SENSOR_TYPE]
    occurences = collections.Counter(sensor_codes)
    if occurences[str(ACCELEROMETER_CODE)] == 0:
        return False
    if occurences[str(GYROSCOPE_CODE)] == 0:
        return False
    if occurences[str(ROTATION_MOTION)] == 0:
        return False
    return True


def check_health(position):
    print("****** HEALTH CHECK *******")
    print("Running... ")
    db_wrist = sqlite3.connect(path + "merged_" + position)
    cursor_wrist = db_wrist.cursor()
    cursor_wrist.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(cursor_wrist.fetchall())
    print(participants)
    for p in participants:
        if "Andrea" in p[0]:
            continue
        for duplicate in p:
            check_exercises_for_participant(db_wrist, duplicate, position)


def check_synchrony():
    db_wrist = sqlite3.connect(path + "merged_wrist")
    db_ankle = sqlite3.connect(path + "merged_ankle")
    cursor_wrist = db_wrist.cursor()
    cursor_wrist.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(cursor_wrist.fetchall())
    for p in participants:
        print("\n")
        print(p[0])
        for code in WORKOUT:
            readings_w = get_participant_readings_for_exercise(db_wrist, p[0], code)
            readings_a = get_participant_readings_for_exercise(db_ankle, p[0], code)
            if readings_a is None or readings_w is None:
                print(EXERCISE_CODES_TO_NAME[code] + " skipped. No Readings available")
                continue
            timestampsW = readings_w[:, READING_TIMESTAMP]
            timestampsA = readings_a[:, READING_TIMESTAMP]
            if abs(int(timestampsA[0]) - int(timestampsW[0])) > 100:
                print(p[0] + " " + EXERCISE_CODES_TO_NAME[code] + ": readings more than 100ms apart")
                print(abs(int(timestampsA[0]) - int(timestampsW[0])))
    db_wrist.commit()
    db_ankle.commit()


###

def prepare_data():
    shutil.rmtree(numpy_exercises_data_path, ignore_errors=True)
    shutil.rmtree(numpy_reps_data_path, ignore_errors=True)
    db_wrist = sqlite3.connect(path + "/merged_wrist")
    db_ankle = sqlite3.connect(path + "/merged_ankle")
    partipant_to_exercises_codes_map = {}
    for code in WORKOUT:
        partipants = get_participants(db_wrist)
        for p in partipants:
            wrist_readings = get_participant_readings_for_exercise(db_wrist, p[0], code)
            if (wrist_readings is None or wrist_readings.size == 0):
                continue
            ankle_readings = get_participant_readings_for_exercise(db_ankle, p[0], code)
            if (ankle_readings is None or ankle_readings.size == 0):
                continue
            if (not were_all_sensors_recorded(ankle_readings) or not were_all_sensors_recorded(wrist_readings)):
                continue
            interpolated_exercise_readings = interpolate_readings(wrist_readings, ankle_readings)
            ex_id = get_exercises_id_for_participant_and_code(db_wrist, p[0], code)
            save_exercise_npy(interpolated_exercise_readings, code, ex_id[0])

            # reps
            single_reps_readings_wrist = get_sub_readings_from_readings_for_wrist(wrist_readings)
            single_reps_readings_ankle = derive_sub_readings_for_ankle_from_wrist(single_reps_readings_wrist,
                                                                                  ankle_readings)
            for i in range(0, min(len(single_reps_readings_wrist), len(single_reps_readings_ankle))):
                interpolated_rep_reading = interpolate_readings(single_reps_readings_wrist[i],
                                                                single_reps_readings_ankle[i])
                save_rep_npy(interpolated_rep_reading, code, ex_id[0], i)
            if p[0] not in partipant_to_exercises_codes_map.keys():
                partipant_to_exercises_codes_map[p[0]] = [ex_id[0]]
            else:
                partipant_to_exercises_codes_map[p[0]].append(ex_id[0])
    with open('file.txt', 'w') as file:
        file.write(json.dumps(partipant_to_exercises_codes_map))  # use `json.loads` to do the reverse


def save_rep_npy(rep_readings, exercise_code, exercise_id, rep):
    exercise_name = EXERCISE_CODES_TO_NAME[exercise_code]
    if not os.path.exists(numpy_reps_data_path):
        os.makedirs(numpy_reps_data_path)
    if not os.path.exists(numpy_reps_data_path + (exercise_name)):
        os.makedirs(numpy_reps_data_path + (exercise_name))
    np.save(numpy_reps_data_path + (exercise_name) + "/" + exercise_name + "_" + str(exercise_id) + "_" + str(rep),
            rep_readings)


def save_exercise_npy(exercise_readings, exercise_code, exercise_id):
    exercise_name = EXERCISE_CODES_TO_NAME[exercise_code]
    if not os.path.exists(numpy_exercises_data_path):
        os.makedirs(numpy_exercises_data_path)
    if not os.path.exists(numpy_exercises_data_path + exercise_name):
        os.makedirs(numpy_exercises_data_path + exercise_name)
    np.save(numpy_exercises_data_path + exercise_name + "/" + exercise_name + "_" + str(exercise_id),
            exercise_readings)


def interpolate_readings(wrist_readings, ankle_readings):
    acc_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(ACCELEROMETER_CODE)]
    gyro_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(GYROSCOPE_CODE)]
    rot_readings_w = wrist_readings[wrist_readings[:, READING_SENSOR_TYPE] == str(ROTATION_MOTION)]

    acc_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(ACCELEROMETER_CODE)]
    gyro_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(GYROSCOPE_CODE)]
    rot_readings_a = ankle_readings[ankle_readings[:, READING_SENSOR_TYPE] == str(ROTATION_MOTION)]

    acc_readings_values_w = extract_sensor_readings_values(acc_readings_w[:, READING_VALUES])
    gyro_readings_values_w = extract_sensor_readings_values(gyro_readings_w[:, READING_VALUES])
    rot_readings_values_w = extract_sensor_readings_values(rot_readings_w[:, READING_VALUES])

    acc_readings_values_a = extract_sensor_readings_values(acc_readings_a[:, READING_VALUES])
    gyro_readings_values_a = extract_sensor_readings_values(gyro_readings_a[:, READING_VALUES])
    rot_readings_values_a = extract_sensor_readings_values(rot_readings_a[:, READING_VALUES])

    timestamps_wrist = extract_timestamps(wrist_readings)
    timestamps_ankle = extract_timestamps(ankle_readings)

    start_timestamp = 0

    end_timestamp = min(np.max(timestamps_wrist), np.max(timestamps_ankle))

    step = 10
    equaly_spaced_apart_timestamps = np.array(list(range(start_timestamp, end_timestamp + 1, step)))
    interpolated_readings = np.zeros((3 * 3 * 2 + 1, equaly_spaced_apart_timestamps.shape[0]))

    values_list = [acc_readings_values_w, gyro_readings_values_w, rot_readings_values_w, acc_readings_values_a,
                   gyro_readings_values_a, rot_readings_values_a]
    time_stamp_list = [extract_timestamps(acc_readings_w), extract_timestamps(gyro_readings_w), extract_timestamps(
        rot_readings_w), extract_timestamps(acc_readings_a), extract_timestamps(gyro_readings_a),
                       extract_timestamps(rot_readings_a)]
    current_indexs = 0
    for i in range(0, len(values_list)):
        original_timestamps = time_stamp_list[i]
        interpolated_x = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 0])
        interpolated_y = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 1])
        interpolated_z = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values_list[i][:, 2])
        interpolated_x = interpolated_x.reshape(interpolated_x.shape[0], 1)
        interpolated_y = interpolated_y.reshape(interpolated_y.shape[0], 1)
        interpolated_z = interpolated_z.reshape(interpolated_z.shape[0], 1)
        concatenate = np.concatenate(
            (interpolated_x.transpose(), interpolated_y.transpose(), interpolated_z.transpose()))
        interpolated_readings[1 + current_indexs * 3: 1 + current_indexs * 3 + 3, :] = concatenate
        current_indexs += 1
    return interpolated_readings


def extract_sensor_readings_values(readings):
    sensor_reading_data = np.zeros([np.shape(readings)[0], 3])
    i = 0
    for reading in readings:
        vals = np.array(reading.split(" ")[0:3])
        vals = vals.astype(np.float)
        sensor_reading_data[i] = vals
        i = i + 1
    return sensor_reading_data


def get_sub_readings_from_readings_for_wrist(readings):
    reps = np.unique(readings[:, READING_REP])
    sub_readings = []
    for rep in reps:
        # select all rows where the rep number equals r
        a = readings[readings[:, READING_REP] == rep]
        sub_readings.append(a)
    return sub_readings


def derive_sub_readings_for_ankle_from_wrist(wrist_rep_readings, ankle_readings):
    ankle_fist_timestamp = ankle_readings[0, READING_TIMESTAMP].astype(np.int64)
    last = 0
    reps = []
    for rep in wrist_rep_readings:
        timestamps = extract_timestamps(rep)
        start = last
        end = last + timestamps[timestamps.shape[0] - 1]
        filtered = ankle_readings[
            start <= ankle_readings[:, READING_TIMESTAMP].astype(np.int64) - ankle_fist_timestamp]
        filtered = filtered[filtered[:, READING_TIMESTAMP].astype(np.int64) - ankle_fist_timestamp <= end]
        if (filtered.size == 0):
            continue
        reps.append(filtered)
        last = end + 1
    return reps


def plot_all_exercises_same_type(db, ex_code):
    cursor_ankle = db.cursor()
    cursor_ankle.execute('SELECT id FROM {tn} WHERE exercise_code = {ec} '.format(tn=EXERCISES_TABLE_NAME, ec=ex_code))
    exs = np.array(cursor_ankle.fetchall())
    plot = None
    for e in exs:
        plot = plot_exercise_from_db(e[0], ex_code)
    plot.show()


def plot_exercise_from_db(exId, exerciseCode, sensorType=ACCELEROMETER_CODE):
    conn = sqlite3.connect(path + "merged_ankle")
    c = conn.cursor()

    c.execute(
        'SELECT * FROM {tn} WHERE exercise_id={exid} AND sensor_type={st}'.format(tn=READINGS_TABLE_NAME,
                                                                                  st=sensorType,
                                                                                  exid=exId))
    table = np.array(c.fetchall())
    if table.size == 0:
        return None

    values = table[:, READING_VALUES]
    # extract reps
    reps = table[:, 6]
    rep_starts = np.zeros([reps.shape[0], 1])
    for i in range(0, reps.shape[0] - 1):
        if reps[i] != reps[i + 1] or i == 0:
            rep_starts[i] = True

    sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensorType]])

    i = 0
    for reading in values:
        vals = np.array(reading.split(" "))
        vals = vals.astype(np.float)
        sensorReadingData[i] = vals
        i = i + 1

    # print(sensorReadingData[:, 0])

    timestamps = table[:, 4].astype("int64")
    timestamps = timestamps - timestamps[0]

    plt.figure()
    plt.suptitle(EXERCISE_CODES_TO_NAME[exerciseCode] + " " + SENSOR_TO_NAME[sensorType] + " " + str(exId), fontsize=13)

    plt.subplot(3, 1, 1)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    plt.ylabel('x')
    addRepSeparators(plt, rep_starts, timestamps)

    plt.plot(timestamps, sensorReadingData[:, 0], 'r-')

    plt.subplot(3, 1, 2)
    plt.ylabel('y')
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    addRepSeparators(plt, rep_starts, timestamps)
    # resampling
    interpol = interpolate(timestamps, sensorReadingData[:, 1])
    plt.plot(interpol['x'], interpol['y'], 'b-')
    # plt.plot(timestamps, sensorReadingData[:, 1], 'b-')
    # plt.plot(timestamps, smooth(sensorReadingData[:, 1], 40), 'b--')

    plt.subplot(3, 1, 3)
    plt.ylabel('z')
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    addRepSeparators(plt, rep_starts, timestamps)
    plt.plot(timestamps, sensorReadingData[:, 2], 'g-')

    return plt


def plot_overlap():
    db_wrist = sqlite3.connect(path + "merged_wrist")
    db_ankle = sqlite3.connect(path + "merged_ankle")

    cursor_wrist = db_wrist.cursor()
    cursor_ankle = db_ankle.cursor()
    cursor_wrist.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(cursor_wrist.fetchall())
    plot = None
    participants = [["Virginia Storni"], ["Max abe"], ["Matteo"], ["Tobi bro"], ["Damian ga"],
                    ["Matt senn"], ["Donato pari"], ["Riccardo rigamonti"],
                    ["Desiree Heller"], ["Karol Wojtas"], ["Renata farkas"]]
    participants = [["Donato pari"]]
    EXERCISES = WORKOUT
    for exercise in EXERCISES:
        for p in participants:
            if not did_participant_perform_exercise(db_wrist, p[0], exercise) or not did_participant_perform_exercise(
                    db_ankle,
                    p[0],
                    exercise):
                continue
            id_wrist = get_exercises_id_for_participant_and_code(db_wrist, p[0], exercise)
            id_ankle = get_exercises_id_for_participant_and_code(db_ankle, p[0], exercise)
            readings_wrist = get_readings_for_exercise(db_wrist, id_wrist, ACCELEROMETER_CODE)
            readings_ankle = get_readings_for_exercise(db_ankle, id_ankle, ACCELEROMETER_CODE)
            if readings_wrist.size == 0 or readings_ankle.size == 0:
                continue
            plt.figure()
            plot = plot_wrist_ankle_overlap(readings_wrist, readings_ankle, exercise, p[0])
    plot.show()


def plot_wrist_ankle_overlap(readings_wrist, readings_ankle, exerciseCode, participant, sensorType=ACCELEROMETER_CODE):
    rep_starts_wrist = extract_reps_start_timestamps(readings_wrist)
    rep_starts_ankle = extract_reps_start_timestamps(readings_ankle)

    values_wrist = readings_wrist[:, READING_VALUES]
    values_ankle = readings_ankle[:, READING_VALUES]

    sensorReadingData_wrist = extract_readings_floats(values_wrist, sensorType)
    sensorReadingData_ankle = extract_readings_floats(values_ankle, sensorType)

    shortest = min(sensorReadingData_ankle.shape[0], sensorReadingData_wrist.shape[0])

    sensorReadingData_ankle = sensorReadingData_ankle[0:shortest, :]
    sensorReadingData_wrist = sensorReadingData_wrist[0:shortest, :]

    timestamps_wrist = extract_timestamps(readings_wrist)[0:shortest]
    timestamps_ankle = extract_timestamps(readings_ankle)[0:shortest]

    plt.suptitle(EXERCISE_CODES_TO_NAME[exerciseCode] + " " + SENSOR_TO_NAME[sensorType] + " " + participant,
                 fontsize=13)

    # plt.subplot(total, 1, index + 1)
    plt.xticks(np.arange(min(timestamps_wrist), max(timestamps_wrist) + 1, 1000))
    plt.ylabel('x')
    addRepSeparators(plt, rep_starts_wrist, timestamps_wrist)
    # addRepSeparators(plt, rep_starts_ankle, timestamps_wrist)

    plt.plot(timestamps_wrist, sensorReadingData_wrist[:, 1], 'r-')
    plt.plot(timestamps_ankle, sensorReadingData_ankle[:, 1], 'b-')
    #
    # plt.subplot(total, 1, index)
    # # plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    # plt.ylabel('y')
    # addRepSeparators(plt, rep_starts_wrist, timestamps_wrist)
    #
    # plt.plot(timestamps_wrist, sensorReadingData_wrist[:, 1], 'r-')
    # plt.plot(timestamps_ankle, sensorReadingData_ankle[:, 1], 'b-')
    #
    # plt.subplot(total, 1, 3)
    # plt.ylabel('z')

    # addRepSeparators(plt, rep_starts_wrist, timestamps_wrist)
    # plt.plot(timestamps_wrist, sensorReadingData_wrist[:, 2], 'r-')
    # plt.plot(timestamps_ankle, sensorReadingData_ankle[:, 2], 'b-')

    return plt


def find_index_of_last_reading_of_last_complete_rep(readings):
    reps = readings[:, READING_REP]
    amax = np.amax(reps)


def extract_timestamps(readings_entries):
    timestamps = readings_entries[:, 4].astype("int64")
    timestamps = timestamps - timestamps[0]
    return timestamps


def extract_readings_floats(values, sensorType):
    sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensorType]])
    i = 0
    for reading in values:
        vals = np.array(reading.split(" "))[0:3]
        vals = vals.astype(np.float)
        sensorReadingData[i] = vals
        i = i + 1
    return sensorReadingData


def extract_reps_start_timestamps(readings_wrist):
    reps = readings_wrist[:, READING_REP]
    rep_starts = np.zeros([reps.shape[0], 1])
    for i in range(0, reps.shape[0] - 1):
        if reps[i] != reps[i + 1] or i == 0:
            rep_starts[i] = True
    return rep_starts


def print_workout_info():
    db_wrist = sqlite3.connect(path + "merged_wrist")
    cursor_wrist = db_wrist.cursor()

    ex_to_reps = {}
    print("Counting reps per exercise...")
    for ex_code in WORKOUT:
        exercise_ids = np.array(cursor_wrist.execute(
            'SELECT id FROM {tn} WHERE exercise_code = {ec} '.format(tn=EXERCISES_TABLE_NAME,
                                                                     ec=ex_code)).fetchall())
        reps_count = 0
        for id in exercise_ids:
            reps = np.array(cursor_wrist.execute(
                'SELECT rep_count FROM {tn} WHERE exercise_id = {id} '.format(tn=READINGS_TABLE_NAME,
                                                                              id=id[0])).fetchall())
            if reps.shape[0] == 0:
                continue
            amax = np.amax(reps)
            reps_count += amax
        ex_to_reps[EXERCISE_CODES_TO_NAME[ex_code]] = reps_count

    for key, value in ex_to_reps.items():
        print(key + " : " + str(value))


def remove_1_2_rep_exercises(position):
    print("\n")
    print("**** REMOVING ONE REP EXERCISES... ***** ")
    print("\n")
    db = sqlite3.connect(path + "merged_" + position)
    cursor = db.cursor()
    for ex_code in WORKOUT:
        exercise_ids = np.array(cursor.execute(
            'SELECT id FROM {tn} WHERE exercise_code = {ec} '.format(tn=EXERCISES_TABLE_NAME,
                                                                     ec=ex_code)).fetchall())
        for id in exercise_ids:
            reps = np.array(cursor.execute(
                'SELECT rep_count FROM {tn} WHERE exercise_id = {id} '.format(tn=READINGS_TABLE_NAME,
                                                                              id=id[0])).fetchall())
            if reps.shape[0] == 0:
                print("Removed " + EXERCISE_CODES_TO_NAME[ex_code] + " " + get_participant_for_exercise_id(db,
                                                                                                           id[
                                                                                                               0]) + " because empty")
                cursor.execute("DELETE FROM {tn} WHERE id={id}".format(tn=EXERCISES_TABLE_NAME, id=id[0]))
                continue
            amax = np.amax(reps)
            if amax < 3:
                print("Removed " + EXERCISE_CODES_TO_NAME[ex_code] + " " + get_participant_for_exercise_id(db,
                                                                                                           id[
                                                                                                               0]) + " because only 1 rep")
                cursor.execute("DELETE FROM {tn} WHERE id={id}".format(tn=EXERCISES_TABLE_NAME, id=id[0]))
                continue
            if amax == 2:
                print("Exercise " + str(id) + "has two reps")

    db.commit()
    cursor.close()


def remove_delay_foot(participant, delay_in_ms):
    print(participant)
    db = sqlite3.connect(path + "merged_ankle")
    for ex in WORKOUT:
        readings = get_participant_readings_for_exercise(db, participant, ex)
        if (readings is not None and readings.size > 0):
            first_timestamp = readings[0, READING_TIMESTAMP].astype("int64")
            cut_timestamp = first_timestamp + delay_in_ms
            cut_id = readings[0, READING_ID]
            for i in range(0, readings.shape[0]):
                if (readings[i, READING_TIMESTAMP].astype("int64") > cut_timestamp):
                    break
                else:
                    cut_id = readings[i, READING_ID]
            if cut_id == readings[0, READING_ID]:
                continue
            c = db.cursor()
            c.execute("DELETE FROM {tn} WHERE id>={first_id} AND id<={cut_id}".format(tn=READINGS_TABLE_NAME,
                                                                                      first_id=readings[0, READING_ID],
                                                                                      cut_id=cut_id))
            db.commit()
            c.close()


def find_rep_duration_in_sec(readings):
    time_stamps = []
    rep_starts = extract_reps_start_timestamps(readings)
    for i in range(0, rep_starts.shape[0]):
        if rep_starts[i][0] == 1:
            time_stamps.append(readings[i, READING_TIMESTAMP].astype(np.int64))
    avg = 0
    for j in range(1, len(time_stamps)):
        print(time_stamps[j] - time_stamps[j - 1])
        avg += (time_stamps[j] - time_stamps[j - 1])
    return int(round(avg / len(time_stamps) / 100))


# dont' use
def shift_reps(db, ex_id, readings, rep_duration):
    current_rep = 1
    first_rep_start = readings[0, READING_TIMESTAMP].astype(np.int64)
    last = readings[readings.shape[0] - 1, READING_TIMESTAMP].astype(np.int64)
    current_start = first_rep_start
    while current_start < last:
        finish = current_start + rep_duration * 100
        c = db.cursor()
        c.execute(
            "UPDATE {tn} SET rep_count='{rep}' WHERE exercise_id = {ex_id} AND timestamp>={t_start} AND timestamp<={t_end} ".format(
                tn=READINGS_TABLE_NAME,
                rep=current_rep,
                ex_id=ex_id[0],
                t_start=current_start,
                t_end=finish))
        current_start = finish
        current_rep += 1
    db.commit()
    c.close()


def do_common_preprocessing():
    for p in SMARTWATCH_POSITIONS:
        print('\n')
        print("********** " + p + " ************")
        print('\n')
        remove_zero_reps(p)
        remove_1_2_rep_exercises(p)
    for p, d in DELAYS.items():
        remove_delay_foot(p, d)
    adjust_reversed_watch_position()


# don't use this
def remove_dirty_reps():
    for p in SMARTWATCH_POSITIONS:
        db = sqlite3.connect(path + "merged_" + p)
        c = db.cursor()
        for participant, reps_range_map in reps_to_keep.items():
            print(participant)
            rep_ranges = reps_to_keep[participant]
            for ex_code, reps_list in rep_ranges.items():
                if reps_list is None:
                    continue
                readings = get_participant_readings_for_exercise(db, participant, ex_code)
                if readings is None or readings.size == 0:
                    continue
                readings_lower = readings[readings[:, READING_REP].astype(np.int32) < reps_list.start]
                readings_upper = readings[readings[:, READING_REP].astype(np.int32) >= reps_list.stop]
                if readings_lower.shape[0] > 0:
                    c.execute(
                        "DELETE FROM {tn} WHERE id>={first_of_lower} AND id<={last_of_lower}".format(
                            tn=READINGS_TABLE_NAME,
                            last_of_lower=
                            readings_lower[
                                readings_lower.shape[
                                    0] - 1, READING_ID],
                            first_of_lower=
                            readings_lower[
                                0, READING_ID]))
                if (readings_upper.shape[0] > 0):
                    c.execute(
                        "DELETE FROM {tn} WHERE id>={first_of_upper} AND id<={last_of_upper}".format(
                            tn=READINGS_TABLE_NAME,
                            last_of_upper=
                            readings_upper[
                                readings_upper.shape[
                                    0] - 1, READING_ID],
                            first_of_upper=
                            readings_upper[
                                0, READING_ID]))

        db.commit()
        c.close()


# split_wrist_ankle_and_merge()
# do_common_preprocessing()
prepare_data()
