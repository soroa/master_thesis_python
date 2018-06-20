import numpy as np
import matplotlib.pyplot as plt

from data_loading import get_reps_data


def normalized(a, axis=2, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# X, y = get_reps_data()
# # normalize features
# acc_before_norm = X[0, :, 1]
# X_normalized = normalized(X, axis=2)
# acc_after_norm = X_normalized[0, :, 1]
# t = range(0,X[0,:,0].shape[0])
# plt.figure()
# plt.plot(t, acc_before_norm, 'r-')
# plt.figure()
# plt.plot(t, acc_after_norm, 'b-')
# plt.show()
# a = 0
