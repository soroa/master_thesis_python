import numpy as np
from keras.engine.saving import load_model

from constants import EXERCISE_NAME_TO_CLASS_LABEL, SPEED_WORKOUT, EXECUTION_WORKOUT
from data_loading import extract_test_data, yaml_loader, extract_test_rep_data


def load_rep_counting_models():
    rep_counting_models = {}
    for key, value in EXERCISE_NAME_TO_CLASS_LABEL.iteritems():
        if key is "Null":
            continue
        rep_counting_models[value] = load_model("models/rep_counting_model_" + key + ".h5")
    return rep_counting_models


if __name__ == "__main__":
    config = yaml_loader("./config_cnn.yaml")
    test_windows = extract_test_data("sensor_readings_jan_wrist", "sensor_readings_jan_ankle", ex_code=EXECUTION_WORKOUT)
    model = load_model('./models/recognition_model_with_null.h5')
    preds = model.predict_classes(test_windows) + 1
    probs = model.predict(test_windows)
    window_length = config.get("data_params")['window_length']
    current = -1
    current_start = 0
    streak = 0
    recognized_exercises = {}
    for i in range(0, len(preds)):
        if preds[i] != current:
            if streak >= 8:
                start_time = int((window_length + (current_start - 1) * 0.20 * window_length) / 10)
                end_time = int((window_length + (i - 1) * 0.20 * window_length) / 10)
                recognized_exercises[current] = [start_time, end_time]
            streak = 0
            current_start = i
            current = preds[i]
        else:
            streak += 1
    ex_to_windows_map = extract_test_rep_data("sensor_readings_test_andrea_wrist", "sensor_readings_test_andrea_ankle",
                                              recognized_exercises, ex_code=EXECUTION_WORKOUT)

    rep_counting_models = load_rep_counting_models()
    for key, value in EXERCISE_NAME_TO_CLASS_LABEL.iteritems():
        if key is "Null":
            continue
        windows = ex_to_windows_map[value]
        model = rep_counting_models[value]
        extra_features = (value - 1) * np.ones((windows.shape[0], 1))
        preds = model.predict([windows, extra_features])
        preds_rounded = 1 - preds.argmax(axis=1)
        print(key)
        print(preds_rounded)