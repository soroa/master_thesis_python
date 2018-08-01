import datetime

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from cnn_train import split_train_test
# X, Y = get_reps_features_data()
# X, Y = get_windowed_exerices_raw_training_data()
# mask = ~np.isnan(X).any(axis=1)
# X = X[mask]
# Y = Y[mask]
#
# X_scaled = preprocessing.scale(X)
#
#
# Y_labels = np.argwhere(Y > 0)[:, 1]
# train_features, test_features, train_labels, test_labels = train_test_split(X_scaled, Y_labels, test_size=0.25,
#                                                                             random_state=42)
from utils import yaml_loader, plot_confusion_matrix

now = datetime.datetime.now()
start_time = now.strftime("%Y-%m-%d %H:%M")


def remove_nans(X, Y):
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    Y = Y[mask]
    return [X, Y]


#### Random forestregressor

tuned_parameters_rf = [{'n_estimators': [5, 10, 25, 50, 100, 1000]}]


def random_forest(train_features, train_labels, test_features, test_labels):
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train_features, train_labels)

    test_predictions = clf.predict(test_features)
    test_error = np.sum(test_predictions != test_labels)
    test_accuracy = (test_labels.shape[0] - test_error) / test_labels.shape[0]
    train_predictions = clf.predict(train_features)
    train_error = np.sum(train_predictions != train_labels)
    train_accuracy = (train_labels.shape[0] - train_error) / train_labels.shape[0]
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write("test_accuracy " + str(test_accuracy) + "\n")
    f.write("train_accuracy " + str(train_accuracy)+ "\n")
    f.close()
    return test_predictions


###
# KNN
#####

tuned_parameters_knn = [{'algorithm': ['auto']},
                        {'n_neighbors': [30, 50, 100, 150, 200]}]


def knn(train_features, train_labels, test_features, test_labels):
    neigh = KNeighborsClassifier(n_neighbors=50)
    neigh.fit(train_features, train_labels)
    test_predictions = neigh.predict(test_features)
    train_predictions = neigh.predict(train_features)
    test_error = np.sum(test_predictions != test_labels)
    test_accuracy = (test_labels.shape[0] - test_error) / test_labels.shape[0]
    train_error = np.sum(train_predictions != train_labels)
    train_accuracy = (train_labels.shape[0] - train_error) / train_labels.shape[0]
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write("test_accuracy " + str(test_accuracy) + "\n")
    f.write("train_accuracy " + str(train_accuracy)+ "\n")
    f.close()
    return test_predictions

# knn()

###
def svc(train_features, train_labels, test_features, test_labels):
    clf = SVC(gamma=0.001, C=0.001, kernel='rbf')
    clf.fit(train_features, train_labels)
    test_predictions = clf.predict(test_features)
    test_error = np.sum(test_predictions != test_labels)
    test_accuracy = (test_labels.shape[0] - test_error) / test_labels.shape[0]
    train_predictions = clf.predict(train_features)
    train_error = np.sum(train_predictions != train_labels)
    train_accuracy = (train_labels.shape[0] - train_error) / train_labels.shape[0]
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write("test_accuracy " + str(test_accuracy) + "\n")
    f.write("train_accuracy " + str(train_accuracy)+ "\n")
    f.close()
    return test_predictions


def lr(train_features, train_labels, test_features, test_labels):
    logreg = linear_model.LogisticRegression(C=10)
    logreg.fit(train_features, train_labels)
    test_predictions = logreg.predict(test_features)
    train_predictions = logreg.predict(train_features)
    test_error = np.sum(test_predictions != test_labels)
    test_accuracy = (test_labels.shape[0] - test_error) / test_labels.shape[0]
    train_error = np.sum(train_predictions != train_labels)
    train_accuracy = (train_labels.shape[0] - train_error) / train_labels.shape[0]
    f = open("./reports/report_" + start_time + ".txt", "a+")
    f.write("test_accuracy " + str(test_accuracy) + "\n")
    f.write("train_accuracy " + str(train_accuracy) + "\n")
    f.close()
    return test_predictions


# lr()

#####
#####
# Cross validation
#####
#####

tuned_parameters_svc = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-2, 1, 10, 100],
                         'C': [0.001, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['rbf,linear'], 'C': [0.001, 0.1, 1, 10, 100, 1000]}]


def gridSearch(classifier, parameters):
    # Set the parameters by cross-validation

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(classifier, parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(train_features, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_labels, clf.predict(test_features)
        print(classification_report(y_true, y_pred))
        print()


#
partipant_to_exercises_codes_map = yaml_loader("./config_cnn.yaml").get('participant_to_ex_code_map')
#
# ids = list(partipant_to_exercises_codes_map.values())
# training_set_size = int(len(ids)*0.90)
# test_set = []
# while len(ids)>training_set_size:
#     ind = random.randint(0,len(ids)+1)
#     del ids[ind]
#
# training_ids = []
# for ids in ids:
#     training_ids += ids


config = yaml_loader("./config_cnn.yaml")

conf_matrix = None
classes = ["Push ups", "Pull ups", "Burpess", "Deadlifts", "Box jumps", "Squats", "Situps", "WB", "KB Press",
           "Thrusters"]
# for p_test in partipant_to_exercises_codes_map.keys():
#     print("Leaving out " + p_test)
#     train_exercise_codes = []
#     for p in partipant_to_exercises_codes_map:
#         if p == p_test:
#             continue
#         train_exercise_codes += (partipant_to_exercises_codes_map[p])
#     train_features, train_labels, test_features, test_labels = get_windowed_exerices_feautres_for_training_data(True,
#                                                                                                                 train_exercise_codes,
#                                                                                                                 config)
#     f = open("./reports/report_" + start_time + ".txt", "a+")
#     f.write("Leave out:  %s\r\n" % p_test)
#     f.close()
#     test_predictions = svc(train_features, train_labels, test_features, test_labels)
#     # c = [np.where(r == 1)[0][0] for r in test_labels]
#     cm = confusion_matrix(test_labels, test_predictions)
#     if conf_matrix is not None:
#         if cm.shape != (10, 10):
#             padded_conf_matrix = np.zeros((10, 10)).astype(np.int32)
#             padded_conf_matrix[0:cm.shape[0], 0:cm.shape[1]] = cm
#             cm = padded_conf_matrix
#         conf_matrix = conf_matrix + cm
#     else:
#         if cm.shape != (10, 10):
#             padded_conf_matrix = np.zeros((10, 10)).astype(np.int32)
#             padded_conf_matrix[0:cm.shape[0], 0:cm.shape[1]] = cm
#             cm = padded_conf_matrix
#         conf_matrix = cm
#
#     plot_confusion_matrix(cm, classes, title=p_test)
#
# plot_confusion_matrix(conf_matrix, classes, title="total_matrix")
