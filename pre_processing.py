import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from constants import *
from plot import addRepSeparators, interpolate

tables = [WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME]

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
        b_cursor = db.cursor()
        for table in tables:
            print(table)
            b_cursor.execute('SELECT * FROM {tn}'.format(tn=table))
            entries = b_cursor.fetchall()  # Returns the results as a list.
            for e in entries:
                if table is READINGS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?,?,?"
                elif table is EXERCISES_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                elif table is WORKOUTS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                first_cursor.execute('INSERT INTO {tn} VALUES({values_placeholder})'.format(tn=table,
                                                                                            values_placeholder=values_placeholder),
                                     e)
        b_cursor.close()

    first_db.commit()
    first_cursor.close()
    os.rename(path + first_name, path + "merged_" + position)


def print_participant_name():
    for name in dbs_names:
        db = sqlite3.connect(path + name)
        b_cursor = db.cursor()
        b_cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
        participants = np.array(b_cursor.fetchall())
        for p in participants:
            print(name + " " + p[0])


def remove_null_reps(position):
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
        reps = get_reps_for_exercise_id(db, id)
        if reps.size == 0:
            print(participant + " did not perform any reps for exercise " + get_exercise_name_for_id(db,
                                                                                                     id) + " on " + position)
    if healthy:
        print(participant + " " + position + " readings are are all there")


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


# check that readings are there


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


def get_reps_for_exercise_id(db, id):
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


def plot_exercise(exId, exerciseCode, sensorType=ACCELEROMETER):
    conn = sqlite3.connect(path+"merged_ankle")
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
