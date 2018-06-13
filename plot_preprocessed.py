import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import os

from constants import *


def plot_reps():
    reps_file_names = os.listdir(numpy_reps_data_path+EXERCISE_CODES_TO_NAME[PUSH_UPS])
    for file in reps_file_names[0:6]:
        rep = np.load(numpy_reps_data_path+EXERCISE_CODES_TO_NAME[PUSH_UPS]+"/"+file)
        plot_rep(rep, "PUSH UPS")
    plt.show()


def plot_rep(rep, exname):

    #acc wrist
    plt.figure()
    plt.suptitle(exname , fontsize=13)
    plt.subplot(6,3 ,1)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc x')
    plt.plot(timestamps, rep[WRIST_ACCEL_X,:], 'r-')

    plt.subplot(6,3 ,2)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc y')
    plt.plot(timestamps, rep[WRIST_ACCEL_Y,:], 'r-')

    plt.subplot(6,3 ,3)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc z')
    plt.plot(timestamps, rep[WRIST_ACCEL_Z,:], 'r-')


    #gyro wrist
    plt.subplot(6,3 ,4)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro x')
    plt.plot(timestamps, rep[WRIST_GYRO_X,:], 'b-')

    plt.subplot(6,3 ,5)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro y')
    plt.plot(timestamps, rep[WRIST_GYRO_Y,:], 'b-')

    plt.subplot(6,3 ,6)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro z')
    plt.plot(timestamps, rep[WRIST_GYRO_Z,:], 'b-')

    #rot wrist
    plt.subplot(6,3 ,7)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot x')
    plt.plot(timestamps, rep[WRIST_ROT_X,:], 'g-')

    plt.subplot(6,3 ,8)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot y')
    plt.plot(timestamps, rep[WRIST_ROT_Y,:], 'g-')

    plt.subplot(6,3 ,9)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot z')
    plt.plot(timestamps, rep[WRIST_ROT_Z,:], 'g-')

    #acc ankle
    plt.subplot(6,3 ,10)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc x')
    plt.plot(timestamps, rep[ANKLE_ACCEL_X,:], 'r-')

    plt.subplot(6,3 ,11)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc y')
    plt.plot(timestamps, rep[ANKLE_ACCEL_Y,:], 'r-')

    plt.subplot(6,3 ,12)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('acc z')
    plt.plot(timestamps, rep[ANKLE_ACCEL_Z,:], 'r-')


    #gyro wrist
    plt.subplot(6,3 ,13)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro x')
    plt.plot(timestamps, rep[ANKLE_GYRO_X,:], 'b-')

    plt.subplot(6,3 ,14)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro y')
    plt.plot(timestamps, rep[ANKLE_GYRO_Y,:], 'b-')

    plt.subplot(6,3 ,15)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('gyro z')
    plt.plot(timestamps, rep[ANKLE_GYRO_Z,:], 'b-')

    #rot wrist
    plt.subplot(6,3 ,16)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot x')
    plt.plot(timestamps, rep[ANKLE_ROT_X,:], 'g-')

    plt.subplot(6,3 ,17)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot y')
    plt.plot(timestamps, rep[ANKLE_ROT_Y,:], 'g-')

    plt.subplot(6,3 ,18)
    timestamps = range(0, 10*rep.shape[1], 10)
    plt.xticks(np.arange(min(timestamps), max(timestamps) + 1, 100))
    plt.ylabel('rot z')
    plt.plot(timestamps, rep[ANKLE_ROT_Z,:], 'g-')

plot_reps()


