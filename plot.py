import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from constants import *

readings_table_name = "sensor_readings"
exercises_table_name = "exercises"
workouts_table_name = "workout_sessions"
sqlite_file = '/Users/mac/Downloads/sensor_readings_matt_donato_corneel_ankle'


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def getExercisesIdsForWorkout(workoutId):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute('SELECT id FROM {tn} WHERE workout_session_id={wid}'.format(tn=exercises_table_name, wid=workoutId))
    ids = np.array(c.fetchall())
    c.execute(
        'SELECT exercise_code FROM {tn} WHERE workout_session_id={wid}'.format(tn=exercises_table_name, wid=workoutId))
    exercise_codes = np.array(c.fetchall())
    return [ids, exercise_codes]


def getAllWorkoutIds():
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute('SELECT id FROM {tn}'.format(tn=workouts_table_name))
    ids = np.array(c.fetchall())
    return ids


def plotExercise(sensorType=ROTATION_MOTION, exerciseId=1, exerciseCode=1):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    c.execute(
        'SELECT * FROM {tn} WHERE sensor_type={st} AND exercise_id={exid}'.format(tn=readings_table_name, st=sensorType,
                                                                                  exid=exerciseId))
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
    plt.suptitle(EXERCISE_CODES_TO_NAME[exerciseCode] + " " + SENSOR_TO_NAME[sensorType], fontsize=13)

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


def addRepSeparators(plot, repsDelimiters, timestamps):
    for i in range(0, repsDelimiters.shape[0]):
        if repsDelimiters[i] == 1:
            plot.axvline(x=timestamps[i])


def plotAllExercisesForSession(sessionId, sensorCode):
    ex_ids, exercise_codes = getExercisesIdsForWorkout(sessionId)
    plt = None

    for id in ex_ids:
        index = np.where(ex_ids == id)
        plt = plotExercise(sensorCode, id[0], exercise_codes[index][0])
        # plt.show()
    plt.show()


def interpolate(x, y):
    step = 10
    print(list(range(0, x[x.shape[0] - 1], step)))
    equaly_spaced_apart_xs = list(range(0, x[x.shape[0] - 1], step))
    print(x.shape)
    print(y.shape)
    interpolated_y = np.interp(equaly_spaced_apart_xs, x, y)
    return {'x': equaly_spaced_apart_xs, 'y': interpolated_y}


ids = getAllWorkoutIds()
plotAllExercisesForSession(ids[ids.shape[0] - 1][0], ACCELEROMETER)
