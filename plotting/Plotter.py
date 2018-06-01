import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from constants import *


class Plotter:

    def interpolate(x, y):
        step = 10
        print(list(range(0, x[x.shape[0] - 1], step)))
        equaly_spaced_apart_xs = list(range(0, x[x.shape[0] - 1], step))
        print(x.shape)
        print(y.shape)
        interpolated_y = np.interp(equaly_spaced_apart_xs, x, y)
        return {'x': equaly_spaced_apart_xs, 'y': interpolated_y}

    def addRepSeparators(plot, repsDelimiters, timestamps):
        for i in range(0, repsDelimiters.shape[0]):
            if repsDelimiters[i] == 1:
                plot.axvline(x=timestamps[i])

    def plot_exercise(exerciseId, exerciseCode, sensorType=ACCELEROMETER_CODE):
        conn = sqlite3.connect(sqlite_file)
        c = conn.cursor()

        c.execute(
            'SELECT * FROM {tn} WHERE sensor_type={st} AND exercise_id={exid}'.format(tn=readings_table_name,
                                                                                      st=sensorType,
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

    def plot_all_exercises_for_code(exercise_code):
        conn = sqlite3.connect(sqlite_file)
        c = conn.cursor()
        c.execute(
            'SELECT *id FROM {tn} WHERE exercise_code={ec}'.format(tn=EXERCISES_TABLE_NAME,
                                                                   ec=exercise_code))
        ex_ids = np.array(c.fetchall())
        plt = None

        for id in ex_ids:
            plt = Plotter.plot_exercise(id, exercise_code)
        plt.show()
