import collections
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from constants import *
from plot import addRepSeparators, interpolate

tables = [WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME]
reversed_ankle_watch_participant_ids = [42, 50, 51, 52, 54, 57, 60, 62, 64, 65, 67, 68, 69]

path = "./dbs2/"
dbs_names = os.listdir(path)
workout_id_count = 0
exercise_id = 0
reading_id = 0


def split_wrist_ankle_and_merge():
    ankle_readings = []
    wrist_readings = []
    for name in dbs_names:
        if "ankle" in name:
            ankle_readings.append(name)
        elif "wrist" in name:
            wrist_readings.append(name)
    print(wrist_readings)
    print(ankle_readings)
    # merge_databases(wrist_readings, "wrist")
    merge_databases(ankle_readings, "ankle")


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
                print("before: " + values_as_string)
                vals_split = np.array(values_as_string.split(" "))[0:3]
                vals_as_float = vals_split.astype(np.float)
                adjusted_values = np.multiply(vals_as_float, np.array([-1, -1, 1]))
                adjust_values_as_string = str(adjusted_values[0]) + " " + str(adjusted_values[1]) + " " + str(
                    adjusted_values[2])
                print("after: " + adjust_values_as_string)
                entry[READING_VALUES] = adjust_values_as_string
                c.execute(
                    "UPDATE {tn} SET [values]='{adjusted_values}' WHERE id={id} ".format(tn=READINGS_TABLE_NAME,
                                                                                         id=entry[READING_ID],
                                                                                         adjusted_values=adjust_values_as_string))
                print(entry[READING_ID])
    db.commit()
    c.close()


def print_participant_name():
    for name in dbs_names:
        db = sqlite3.connect(path + name)
        b_cursor = db.cursor()
        b_cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
        participants = np.array(b_cursor.fetchall())
        for p in participants:
            print(name + " " + p[0])


def remove_zero_reps(position):
    db = sqlite3.connect(path + "merged_" + position)
    cursor = db.cursor()
    cursor.execute('DELETE FROM {tn} WHERE rep_count=0'.format(tn=READINGS_TABLE_NAME))
    db.commit()
    cursor.close()

def remove_Andrea_Soro_workouts(position):
    db = sqlite3.connect(path + "merged_" + position)
    cursor = db.cursor()
    cursor.execute("DELETE FROM {tn} WHERE participant = 'Andrea Soro".format(tn=WORKOUTS_TABLE_NAME))
    db.commit()
    cursor.close()


def find_duplicates_workout(position):
    db = sqlite3.connect(path + "merged_" + position)
    cursor = db.cursor()
    cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = (cursor.fetchall())
    for p in participants:
        if participants.count(p) > 1:
            print("There are duplicate workouts for " + p)

    db.commit()
    cursor.close()


def check_exercises_for_participant(db, participant, position):
    ex_codes = get_exercise_codes_for_participants(db, participant)
    healthy = True
    for code in WORKOUT:
        if code not in ex_codes:
            healthy = False
            print(participant + " did not perform " + EXERCISE_CODES_TO_NAME[code] + " on " + position)
    if healthy:
        print(participant + " " + position + " exercises are ok ")
    check_readings_for_participant(db, participant, position)
    # check that readings are there


def check_readings_for_participant(db, participant, position):
    ex_ids = get_exercises_ids_for_participants(db, participant)
    healthy = True
    for id in ex_ids:
        readings = get_readings_for_exercise(db, id)
        interpolate_readings(readings)
        if readings.size == 0:
            print(participant + "  readings are missing for exercise " + get_exercise_name_for_id(db,
                                                                                                  id) + " on " + position)
        else:
            check_all_sensors_were_recorded(readings, participant, position, id)
    if healthy:
        print(participant + " " + position + " readings are there")


def check_all_sensors_were_recorded(readings, participant, position, ex_id):
    sensor_codes = readings[:, READING_SENSOR_TYPE]
    occurences = collections.Counter(sensor_codes)
    print(occurences)
    if occurences[str(ACCELEROMETER_CODE)] == 0:
        print(participant + " " + position + " acc data is missing for ex_id " + ex_id)
    if occurences[str(GYROSCOPE_CODE)] == 0:
        print(participant + " " + position + " gyro data is missing for ex_id " + ex_id)
    if occurences[str(ROTATION_MOTION)] == 0:
        print(participant + " " + position + " rot data is missing for ex_id " + ex_id)


def check_health():
    db_wrist = sqlite3.connect(path + "merged_wrist")
    cursor_wrist = db_wrist.cursor()
    cursor_wrist.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(cursor_wrist.fetchall())
    print(participants)
    for p in participants:
        if "Andrea" in p[0]:
            continue
        for duplicate in p:
            check_exercises_for_participant(db_wrist, duplicate, "wrist")


def check_synchrony():
    db_wrist = sqlite3.connect("./dbs/merged_wrist")
    db_ankle = sqlite3.connect("./dbs/merged_ankle")
    cursor_wrist = db_wrist.cursor()
    cursor_ankle = db_ankle.cursor()
    cursor_wrist.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
    participants = np.array(cursor_wrist.fetchall())
    for p in participants:
        if "Andrea" in p[0]:
            continue
        ids_exer_wrist = get_exercises_ids_for_participants(db_wrist, p[0])
        ids_exer_ankle = get_exercises_ids_for_participants(db_ankle, p[0])
        for i in range(0, ids_exer_wrist.shape[0]):
            print(p[0])
            timestampsW = np.array(cursor_wrist.execute(
                'SELECT timestamp FROM {tn} WHERE exercise_id={ex_id}'.format(tn=READINGS_TABLE_NAME,
                                                                              ex_id=ids_exer_wrist[i, 0])).fetchall())

            timestampsA = np.array(cursor_ankle.execute(
                'SELECT timestamp FROM {tn} WHERE exercise_id={ex_id}'.format(tn=READINGS_TABLE_NAME,
                                                                              ex_id=ids_exer_ankle[i, 0])).fetchall())
            if timestampsW.size == 0 or timestampsA.size == 0:
                print("Exercise Wrist not completed: id " + str(ids_exer_wrist[i, 0]))
                continue
            if timestampsA.size == 0:
                print("Exercise Ankle not completed: id " + str(ids_exer_wrist[i, 0]))
                continue

            print("exercise: " + str(ids_exer_wrist[i, 0]))
            print(timestampsA[0] - timestampsW[0])

    db_wrist.commit()
    db_ankle.commit()
    cursor_ankle.close()
    cursor_wrist.close()


###

def prepare_data():
    remove_zero_reps("wrist")
    db = sqlite3.connect(path + "/merged_wrist")
    cursor = db.cursor()
    for code in WORKOUT:
        cursor.execute('SELECT id FROM {tn} WHERE exercise_code = {code} '.format(tn=EXERCISES_TABLE_NAME, code=code))
        exercises_ids = np.array(cursor.fetchall())
        for id in exercises_ids:
            exercise_readings = get_readings_for_exercise(db, id)
            single_reps_readings = get_sub_readings_from_readings(exercise_readings)
            for rep in single_reps_readings:
                interpolated_rep_reading = interpolate_readings(rep)


def interpolate_readings(readings):
    acc_readings = readings[readings[:, READING_SENSOR_TYPE] == str(ACCELEROMETER_CODE)]
    gyro_readings = readings[readings[:, READING_SENSOR_TYPE] == str(GYROSCOPE_CODE)]
    rot_readings = readings[readings[:, READING_SENSOR_TYPE] == str(ROTATION_MOTION)]

    acc_readings_values = extract_sensor_readings_values(acc_readings[:, READING_VALUES])
    gyro_readings_values = extract_sensor_readings_values(gyro_readings[:, READING_VALUES])
    rot_readings_values = extract_sensor_readings_values(rot_readings[:, READING_VALUES])

    start_timestamp = max(
        [acc_readings[0, READING_TIMESTAMP].astype("int64"), gyro_readings[0, READING_TIMESTAMP].astype("int64"),
         rot_readings[0, READING_TIMESTAMP].astype("int64")])

    last_timestamp = min([acc_readings[acc_readings.shape[0] - 1, READING_TIMESTAMP].astype("int64"),
                          gyro_readings[gyro_readings.shape[0] - 1, READING_TIMESTAMP].astype("int64"),
                          rot_readings[rot_readings.shape[0] - 1, READING_TIMESTAMP].astype("int64")])

    step = 10
    equaly_spaced_apart_timestamps = np.array(list(range(start_timestamp, last_timestamp + 1, step)))
    interpolated_readings = np.zeros((3 * 3 + 1, equaly_spaced_apart_timestamps.shape[0]))

    values_list = [acc_readings_values, gyro_readings_values, rot_readings_values]
    current_indexs = 0
    for values in values_list:
        if values is acc_readings_values:
            original_timestamps = acc_readings[:, READING_TIMESTAMP].astype("int64")
        elif values is gyro_readings_values:
            original_timestamps = gyro_readings[:, READING_TIMESTAMP].astype("int64")
        else:
            original_timestamps = rot_readings[:, READING_TIMESTAMP].astype("int64")

        interpolated_x = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values[:, 0])
        interpolated_y = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values[:, 1])
        interpolated_z = np.interp(equaly_spaced_apart_timestamps, original_timestamps,
                                   values[:, 2])
        interpolated_x = interpolated_x.reshape(interpolated_x.shape[0], 1)
        interpolated_y = interpolated_y.reshape(interpolated_y.shape[0], 1)
        interpolated_z = interpolated_z.reshape(interpolated_z.shape[0], 1)
        concatenate = np.concatenate(
            (interpolated_x.transpose(), interpolated_y.transpose(), interpolated_z.transpose()))
        interpolated_readings[1 + current_indexs*3: 1 + current_indexs*3 + 3, :] = concatenate
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


def get_sub_readings_from_readings(readings):
    reps = np.unique(readings[:, READING_REP])
    sub_readings = []
    for rep in reps:
        # select all rows where the rep number equals r
        a = readings[readings[:, READING_REP] == rep]
        sub_readings.append(a)
    return sub_readings


####

def get_exercises_ids_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                     id=workout_id[0, 0]))
    exercises_ids = cursor.fetchall()
    return np.array(exercises_ids)


def get_exercises_ids_for_workout_id(workout_id, db):
    c = db.cursor()
    c.execute('SELECT id FROM {tn} WHERE workout_session_id={wid}'.format(tn=EXERCISES_TABLE_NAME, wid=workout_id))
    ids = np.array(c.fetchall())
    return ids


def get_exercise_codes_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                                id=workout_id[0, 0]))
    exercise_codes = cursor.fetchall()
    return np.array(exercise_codes)


def get_readings_for_exercise(db, id):
    cursor = db.cursor()
    cursor.execute(
        'SELECT * FROM {tn} WHERE exercise_id = {exid}'.format(tn=READINGS_TABLE_NAME,
                                                               exid=id[0]))
    return np.array(cursor.fetchall())


def get_exercise_name_for_id(db, id):
    cursor = db.cursor()
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                id=id[0]))
    return EXERCISE_CODES_TO_NAME[np.array(cursor.fetchall())[0, 0]]


def plot_all_exercises_same_type():
    db_ankle = sqlite3.connect(path + "/merged_ankle")
    cursor_ankle = db_ankle.cursor()
    ex_code = KETTLEBELL_PRESS
    cursor_ankle.execute('SELECT id FROM {tn} WHERE exercise_code = {ec} '.format(tn=EXERCISES_TABLE_NAME, ec=ex_code))
    exs = np.array(cursor_ankle.fetchall())
    plot = None
    for e in exs:
        plot = plot_exercise(e[0], ex_code)
    plot.show()


def plot_exercise(exId, exerciseCode, sensorType=ACCELEROMETER_CODE):
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

    # print(rep_starts)

    sensorReadingData = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensorType]])

    i = 0
    for reading in values:
        vals = np.array(reading.split(" "))
        vals = vals.astype(np.float)
        sensorReadingData[i] = vals
        i = i + 1

    # print(sensorReadingData[:, 0])
    plt.figure()

    timestamps = table[:, 4].astype("int64")
    timestamps = timestamps - timestamps[0]
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

# check_health()
# find_duplicates_workout("ankle")
# remove_null_reps("wrist")
plot_all_exercises_same_type()
# remove_null_reps("ankle")
# split_wrist_ankle_and_merge()
# adjust_reversed_watch_position()
# prepare_data()
