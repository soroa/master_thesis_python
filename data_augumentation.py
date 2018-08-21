import numpy as np

from constants import SENSOR_CHANNELS


def rescale(arr, factor=2):
    n = len(arr)
    return np.interp(np.linspace(0, n, factor * n + 1), np.arange(n), arr)


def stretch_compress_windows(windows, scaling_factors=None):
    if scaling_factors is None:
        scaling_factors = [0.60, 0.80, 1, 1.20, 1.5]
    rescaled_windows = []
    for w in windows:
        for factor in scaling_factors:
            single_channel_rescaled_list = []
            for channel in SENSOR_CHANNELS:
                single_channel_rescaled_list.append(rescale(w[channel, :], factor))
            rescaled_windows.append(np.asarray(single_channel_rescaled_list))
    return rescaled_windows

#
# def plot(a):
#     n = len(a)
#     print(range(0, n))
#     print(a)
#     plt.plot(np.arange(0, n), a)
#     print(a.shape)

# x = np.arange(0, 2*np.pi,0.1)
# print(x)
# plt.plot(x, np.sin(x))


# a = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0])
# t = np.arange(0, 2*np.pi,0.1)
# a = np.sin(t)
# b = a * 2
# ar = [a]
# sw = stretch_compress_windows(ar)
# plot(a)
# for s in sw:
#     plot(s)
# plt.show()
