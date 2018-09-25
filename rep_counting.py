import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.stats import stats
from sklearn.decomposition import PCA

from constants import CLASS_LABEL_TO_AVERAGE_REP_DURATION
from data_loading import get_exercise_readings, calculate_longest_and_shortest_rep_per_exercise


def butter_band(x):
    return butter_bandpass_filter(x, lowcut, highcut, fs, order=6)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_stats_for_k(a, k):
    consecutives_ks = 0
    map = {}
    i = -1
    for point in a:
        i += 1
        if point == k:
            consecutives_ks += 1
            if i == len(a) - 1:
                # we reach the end
                if consecutives_ks in map.keys():
                    map[consecutives_ks] += 1
                else:
                    map[consecutives_ks] = 1
        else:
            if consecutives_ks != 0:
                if consecutives_ks in map.keys():
                    map[consecutives_ks] += 1
                else:
                    map[consecutives_ks] = 1
                consecutives_ks = 0
    import operator

    mode = max(map.iteritems(), key=operator.itemgetter(1))[0]
    min_k = min(map.iterkeys())
    max_k = max(map.iterkeys())
    return {"min": min_k, "mode": mode, "max": max_k}


def count_predicted_reps(preds):
    slim = []
    consecutive_ones = 0
    for p in preds:
        if p == 0:
            consecutive_ones = 0
        if p == 1:
            consecutive_ones += 1
            if consecutive_ones == 3:
                slim.append(1)

    return np.sum(slim)


def get_sum_by_contraction(sequence):
    sum = 0
    previous = -1
    for s in sequence:
        if s == 1 and previous != 1:
            sum += 1
        previous = s
    return sum


def get_right_side_zeros(preds, i):
    zeros = 0
    for j in range(i, len(preds)):
        if 0 == preds[j]:
            zeros += 1
    return zeros


def get_left_side_zeros(preds, i):
    zeros = 0
    for j in range(i, -1, -1):
        if 0 == preds[j]:
            zeros += 1
        elif preds[j] and zeros > 0:
            break
    return zeros


def count_predicted_reps2(preds):
    if not 1 in preds:
        return 0
    slim = []
    consecutive_ones = 0
    isrep = False
    mode_1 = get_stats_for_k(preds, 1)["mode"]
    for i in range(0, len(preds)):
        if preds[i] == 0:
            isrep = False
            consecutive_ones = 0
        if preds[i] == 1:
            if isrep:
                preds[i] = 2
                continue
            consecutive_ones += 1
            if consecutive_ones > mode_1 * 0.7:
                for j in range(0, 3):
                    preds[i - j] = 2
                isrep = True
                slim.append(1)
    intrarep_mode = get_stats_for_k(preds, 0)["mode"]

    # second run
    consecutive_ones = 0
    for i in range(0, len(preds)):
        if preds[i] == 0:
            consecutive_ones = 0
        if preds[i] == 1:
            if i == (len(preds) - 1) or preds[i + 1] == 0:
                left = get_left_side_zeros(preds, i)
                right = get_right_side_zeros(preds, i)
                if (intrarep_mode - left > intrarep_mode / 3) or (intrarep_mode - right > intrarep_mode / 3):
                    preds[i] = 0
                    if i > 0:
                        preds[i - 1] = 0
                    continue
                else:
                    if (consecutive_ones >= mode_1 / 3):
                        slim.append(1)
            consecutive_ones += 1

    return np.sum(slim)


def smoothing(sequence, min_sequence_length):
    window_length = min_sequence_length * 2 - 1
    if sequence.shape < window_length:
        return sequence
    new_sequence = np.asarray([])
    for i in range(0, sequence.shape[0] - window_length):
        prevailing_number = stats.mode(sequence[i:i + window_length])[0][0]
        new_sequence = np.append(new_sequence,prevailing_number)
    return new_sequence.astype(np.int32)


def count_real_reps(truth):
    slim = []
    for t in truth:
        if len(slim) == 0 or slim[-1] != t:
            slim.append(t)
    return np.sum(slim)


def rep_counting_model_pre_processing(X, padding=0):
    max_length = -1
    for sequence in X:
        if len(sequence) > max_length:
            max_length = len(sequence)
    shape = (len(X), max_length)
    if padding == 0:
        padded_X = np.zeros((shape))
    else:
        padded_X = np.ones((shape)) * padding
    for i in range(0, len(X)):
        padded_X[i] = X[i]
    return padded_X


# def rep_counting_model_with_ml(X_train, y_train, X_test, y_test):


if __name__ == "__main__":

    # Sample rate and desired cutoff frequencies (in Hz).

    # Elliptical bandpass filter (0.15-11Hz)
    # Subtract mean from numpy_data
    # Apply pCA and project onto first PC
    l = calculate_longest_and_shortest_rep_per_exercise()

    fs = 100.0
    lowcut = 0.15
    highcut = 11.0

    X, Y, reps_count = get_exercise_readings()
    errors = 0
    errors_ex_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tot_reps = 0
    for i in range(0, len(X)):
        x = X[i]
        # Filter a noisy signal.
        T = len(x) / fs
        nsamples = len(x)
        t = np.linspace(0, T, nsamples, endpoint=False)
        a = 0.02
        x = np.transpose(x)
        x = np.nan_to_num(x)
        x = x - x.mean(axis=0)
        y = np.apply_along_axis(butter_band, 1, x)
        pca = PCA(n_components=1, svd_solver='auto')
        y_t = pca.fit_transform(y)
        y_t = y_t.reshape((y_t.shape[0]))
        # y_t = np.linalg.norm(y[:,0:3], axis=1)
        avg_duration = CLASS_LABEL_TO_AVERAGE_REP_DURATION[Y[i] - 1]
        # peakind = signal.find_peaks_cwt(y_t.reshape((y_t.shape[0])), np.arange(1, avg_duration + 50, 1))
        peakind = signal.find_peaks_cwt(y_t.reshape((y_t.shape[0])), np.arange(1, avg_duration + 50, 1))

        err = abs(peakind.shape[0] - reps_count[i])
        # if err > 0:
        #     for i in range(0, peakind.shape[0] - 1):
        #         if i == 0:
        #             dist_to_prev = peakind[i]
        #         else:
        #             dist_to_prev = peakind[i] - peakind[i - 1]
        #         if (dist_to_prev < avg_duration - 1000 or dist_to_prev > avg_duration + 100):
        #             if i == peakind.shape[0] - 1:
        #                 peakind = peakind[0:i]
        #             else:
        #                 peakind = np.append(peakind[0:i],peakind[i + 1:])
        #             err = abs(peakind.shape[0] - reps_count[i])
        errors += err
        errors_ex_type[Y[i] - 1] += err
        tot_reps += (reps_count[i])
        print(str(peakind.shape[0]) + " vs " + str(reps_count[i]))
    print("errors : " + str(errors))
    print("error percentage " + str(errors / tot_reps))
    print(errors_ex_type)
