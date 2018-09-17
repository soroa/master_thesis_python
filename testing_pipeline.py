import numpy as np
from keras.engine.saving import load_model

from cnn_train import TrainingRepCountingParameters
from constants import EXERCISE_NAME_TO_CLASS_LABEL, EXECUTION_WORKOUT, EXERCISE_CLASS_LABEL_TO_NAME, NULL_CLASS
from data_loading import extract_test_data, yaml_loader, extract_test_rep_data
from rep_counting import count_predicted_reps2


class RecognizedExercise:
    reps = 0

    def __init__(self, ex_code, start_time, end_time):
        self.ex_code = ex_code
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return EXERCISE_CLASS_LABEL_TO_NAME[self.ex_code]

    def __repr__(self):
        return "RecognizedExercise({}, {}, {}, {})".format(self.ex_code, self.start_time, self.end_time,
                                                           self.get_duration())

    def get_duration(self):
        return self.end_time - self.start_time

    def set_windows(self, windows):
        self.windows = windows


def load_rep_counting_models():
    rep_counting_models = {}
    for key, value in EXERCISE_NAME_TO_CLASS_LABEL.iteritems():
        if key == "Null":
            continue
        rep_counting_models[value] = load_model("models/rep_counting_model_" + key + ".h5")
    return rep_counting_models


def add_timeline(preds, windows_length, shift):
    ret = np.zeros((2, preds.shape[0]))
    ret[0] = preds
    timeline = np.arange(start=0, stop=ret.shape[1] * windows_length * shift, step=(windows_length * shift))
    ret[1] = timeline
    return ret


def is_exercise(pred):
    return not pred is EXERCISE_NAME_TO_CLASS_LABEL["Null"]


def prediction_smoothing(preds):
    pass


def init_best_rep_counting_models_params():
    ex_to_rep_traning_model_params = {}
    ex_to_rep_traning_model_params["Crunches"] = TrainingRepCountingParameters(exercise="Crunches", window_length=200,
                                                                               window_step_slide=0.10)
    ex_to_rep_traning_model_params["KB Press"] = TrainingRepCountingParameters(exercise="Kb Press", window_length=150,
                                                                               window_step_slide=0.10)
    ex_to_rep_traning_model_params["Wall balls"] = TrainingRepCountingParameters(exercise="Wall balls",
                                                                                 window_length=200,
                                                                                 window_step_slide=0.10)
    ex_to_rep_traning_model_params["Push ups"] = TrainingRepCountingParameters("Push ups", window_length=120,
                                                                               window_step_slide=0.05)
    ex_to_rep_traning_model_params["Dead lifts"] = TrainingRepCountingParameters("Dead lifts", window_length=200,
                                                                                 window_step_slide=0.10)
    ex_to_rep_traning_model_params["Burpees"] = TrainingRepCountingParameters("Burpees", window_length=250,
                                                                              window_step_slide=0.10)
    ex_to_rep_traning_model_params["Squats"] = TrainingRepCountingParameters("Squats", window_length=150,
                                                                             window_step_slide=0.10)
    ex_to_rep_traning_model_params["KB Squat press"] = TrainingRepCountingParameters("Kb Squat press",
                                                                                     window_length=150,
                                                                                     window_step_slide=0.05)
    ex_to_rep_traning_model_params["Box jumps"] = TrainingRepCountingParameters("Box Jumps", window_length=200,
                                                                                window_step_slide=0.10)
    ex_to_rep_traning_model_params["Pull ups"] = TrainingRepCountingParameters("Pull ups", window_length=200,
                                                                               window_step_slide=0.10)
    return ex_to_rep_traning_model_params


if __name__ == "__main__":

    # for name in ["andrea", "jan", "starkaor", "desi", "simon"]:
    for name in ["push1", "push2", "push3"]:
        print("")
        print("")
        print(name)
        print("")
        print("")
        config = yaml_loader("./config_cnn.yaml")
        wrist_db_file = "sensor_readings_test_" + name + "_wrist"
        ankle_db_file = "sensor_readings_test_" + name + "_ankle"
        window = 3000
        step = 0.10
        rep_counting_models = load_rep_counting_models()
        augmentation = False
        test_windows = extract_test_data(wrist_db_file, ankle_db_file,
                                         ex_code=NULL_CLASS, window=window, step=step)
        model = load_model('./models/recognition_model_with_null.h5')
        preds = model.predict_classes(test_windows) + 1
        # preds = add_timeline(preds, window, step )
        probs = model.predict(test_windows)
        window_length = config.get("data_params")['window_length']
        model_params = init_best_rep_counting_models_params()
        current_ex_code = -1
        current_start = 0
        streak = 0
        recognized_exercises = []
        for i in range(0, len(preds)):
            if preds[i] == 11:  # null class
                continue
            if preds[i] != current_ex_code:
                if current_ex_code != -1:
                    start_time = int((window_length + (current_start - 1) * step * window_length) / 10)
                    end_time = int((window_length + (i - 1) * step * window_length) / 10)
                    min_one_rep_length = model_params[EXERCISE_CLASS_LABEL_TO_NAME[current_ex_code]].window_length
                    if (end_time - start_time) >= min_one_rep_length:
                        recognized_exercises.append(
                            RecognizedExercise(current_ex_code, start_time=start_time, end_time=end_time))
                streak = 0
                current_start = i
                current_ex_code = preds[i]
            else:
                streak += 1

        recognized_exercises = extract_test_rep_data(wrist_db_file, ankle_db_file,
                                                     recognized_exercises,
                                                     model_params=model_params,
                                                     ex_code=NULL_CLASS,
                                                     augmentation=augmentation, step=0.05)

        exercises = None

        for rec_ex in recognized_exercises:
            if rec_ex.ex_code != 1:
                continue
            model = rep_counting_models[rec_ex.ex_code]
            preds = model.predict([rec_ex.windows])
            preds_rounded = 1 - preds.argmax(axis=1)
            print(EXERCISE_CLASS_LABEL_TO_NAME[rec_ex.ex_code])
            print(str(rec_ex.get_duration()))
            print(preds_rounded)
            print(count_predicted_reps2(preds_rounded))
