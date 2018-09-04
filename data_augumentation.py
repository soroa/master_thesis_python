import numpy as np

from constants import SENSOR_CHANNELS


def rescale(arr, factor=0.5):
    n = len(arr)
    return np.interp(np.linspace(0, n, int(factor * n + 1)), np.arange(n), arr) * (1 / factor)


def stretch_compress_exercise_readings(exercise_reading):
    scaling_factors = [1]
    rescaled_versions = []
    for factor in scaling_factors:
        single_channel_rescaled_list = []
        for channel in SENSOR_CHANNELS:
            rescale1 = rescale(exercise_reading[channel, :], factor)
            single_channel_rescaled_list.append(rescale1)
        rescaled_versions.append(np.asarray(single_channel_rescaled_list))
    return rescaled_versions
