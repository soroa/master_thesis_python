import datetime
import pickle

import numpy as np
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class Result:
    def __init__(self, name):
        self.name = name
        self.directory_name = "/final_results_objects"

    def save_result_object(self):
        # if not os.path.exists(self.directory_name):
        #     os.makedirs(self.directory_name)
        file = open("./" + self.directory_name + "/" + self.name, "w+")
        now = datetime.datetime.now()
        self.last_modified = now.strftime("%Y-%m-%d %H:%M")
        pickle.dump(self, file)
        file.close()


class Result8020(Result):

    def __init__(self, name):
        Result.__init__(self, name)

    def set_result(self, value):
        self.result = value
        self.save_result_object()


class CVResult(Result):

    def __init__(self, name):
        Result.__init__(self, name)
        self.testing_accuracies = []
        self.truth_predicted_values_tuples = []

    def set_results(self, truth_values, predicted_values, testing_accuracy):
        print(truth_values)
        print(predicted_values)
        print(testing_accuracy)
        self.truth_predicted_values_tuples.append((truth_values, predicted_values))
        print(self.truth_predicted_values_tuples)
        self.testing_accuracies.append(testing_accuracy)
        self.save_result_object()

    def get_number_of_subjects(self):
        return len(self.testing_accuracies)


    def get_box_plot_data(self):
        accuracies_per_class = {}
        for subject in self.truth_predicted_values_tuples:
            classes = np.unique(subject[0])
            for cl in classes:
                indexes = np.where(subject[0] == cl)
                accuracy = accuracy_score(subject[0][indexes], subject[1][indexes])
                if cl not in accuracies_per_class.keys():
                    accuracies_per_class[cl] = []
                accuracies_per_class[cl].append(accuracy)
        return accuracies_per_class



        # c = [np.where(r == 1)[0][0] for r in y_test]
        # cm = confusion_matrix(c, prediction_classes)
        # if cm.shape != (10, 10):
        #     padded_conf_matrix = np.zeros((10, 10)).astype(np.int32)
        #     padded_conf_matrix[0:cm.shape[0], 0:cm.shape[1]] = cm
        #     cm = padded_conf_matrix
        # classes = ["Push ups", "Pull ups", "Burpess", "Deadlifts", "Box jumps", "Squats", "Situps", "WB", "KB Press",
        #            "Thrusters"]
        # if config.get('save_confusion_matrices'):
        #     plot_confusion_matrix(cm, classes, title=left_out_participant)
