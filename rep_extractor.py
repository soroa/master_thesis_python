import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from constants import *

readings_table_name = "sensor_readings"
exercises_table_name = "exercises"
workouts_table_name = "workout_sessions"
sqlite_file = '/Users/mac/Downloads/sensor_readings_mikey_wrist'


def extract_data(workout_id):
    plot = None
    for ex_code in WORKOUT:
        ex_ids = get_exercises_for_ex_code(ex_code, workout_id)
        for id in ex_ids:
            readings = get_readings_for_ex_id(id[0])
            rep_readings = get_sub_readings_from_readings(readings)
            for rep in rep_readings:
                plot = plot_readings(rep)
            break
        break
    plot.show()

def get_exercises_for_ex_code(ex_code, workout_id):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute(
        'SELECT id FROM {tn} WHERE exercise_code={ex_code} AND workout_session_id={workout_id}'.format(tn=exercises_table_name,
                                                                                               ex_code=ex_code,
                                                                                               workout_id=workout_id))
    return np.array(c.fetchall())


def get_readings_for_ex_id(ex_id, sens_type=ACCELEROMETER):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute('SELECT * FROM {tn} WHERE exercise_id={ex_id} AND sensor_type={sens_type}'.format(tn=readings_table_name,
                                                                                                ex_id=ex_id,
                                                                                                sens_type=sens_type))
    return np.array(c.fetchall())


def get_sub_readings_from_readings(readings):
    reps = np.unique(readings[:, READING_REP])
    sub_readings = []
    for rep in reps:
        # select all rows where the rep number equals r
        a = readings[readings[:, READING_REP] == rep]
        sub_readings.append(a)
    return sub_readings


def plot_readings(readings, sensorType=ACCELEROMETER):
    values = readings[:, READING_VALUES]
    # extract reps

    reps = readings[:, READING_REP]
    rep_starts = np.zeros([reps.shape[0], 1])
    for i in range(0, reps.shape[0] - 1):
        if reps[i] != reps[i + 1] or i == 0:
            rep_starts[i] = True

    sensor_reading_data = np.zeros([np.shape(values)[0], SENSOR_TO_VAR_COUNT[sensorType]])

    i = 0
    for reading in values:
        vals = np.array(reading.split(" "))
        vals = vals.astype(np.float)
        sensor_reading_data[i] = vals
        i = i + 1

    # print(sensor_reading_data[:, 0])
    plt.figure()

    timestamps = readings[:, 4].astype("int64")
    timestamps = timestamps - timestamps[0]

    # plt.suptitle(EXERCISE_CODES_TO_NAME[exerciseCode] + " " + SENSOR_TO_NAME[sensorType], fontsize=13)

    plt.subplot(3, 1, 1)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    plt.ylabel('x')
    plt.plot(timestamps, sensor_reading_data[:, 0], 'r-')

    plt.subplot(3, 1, 2)
    plt.ylabel('y')
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    plt.plot(timestamps, sensor_reading_data[:, 1], 'b-')
    # plt.plot(timestamps, smooth(sensor_reading_data[:, 1], 40), 'b--')

    plt.subplot(3, 1, 3)
    plt.ylabel('z')
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 1000))
    plt.plot(timestamps, sensor_reading_data[:, 2], 'g-')

    return plt

extract_data(55)
