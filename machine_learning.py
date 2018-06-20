import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from data_loading import get_reps_data, get_reps_features_data
from sklearn.model_selection import train_test_split, GridSearchCV


X, Y = get_reps_features_data()
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
Y = Y[mask]

print(np.argwhere(np.isnan(X)))
X_scaled = preprocessing.scale(X)
Y_labels = np.argwhere(Y>0)[:,1]
train_features, test_features, train_labels, test_labels = train_test_split(X_scaled, Y_labels, test_size=0.25,
                                                                            random_state=42)

#### Random forestregressor

def random_forest():
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train_features, train_labels)

    predictions = clf.predict(test_features)
    errors = np.sum(abs(predictions - test_labels))
    errors = np.sum(abs(predictions - test_labels))
    accuracy = (test_labels.shape[0] - errors) / test_labels.shape[0]
    print("Random Forests")
    print("errors " + str(errors))
    print("accuracy " + str(accuracy))
    print("\n")


###
# KNN
#####
def knn():
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_features, train_labels)
    predictions = neigh.predict(test_features)
    errors = np.sum(abs(predictions - test_labels))
    accuracy = (test_labels.shape[0] - errors) / test_labels.shape[0]
    print("KNN")
    print("errors " + str(errors))
    print("accuracy " + str(accuracy))
    print("\n")

###
def svc():
    clf = SVC(gamma=0.0001, C=1000, kernel='rbf')
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    errors = np.sum(abs(predictions - test_labels))
    accuracy = (test_labels.shape[0]-errors)/test_labels.shape[0]
    print("SVC")
    print("errors " + str(errors))
    print("accuracy " + str(accuracy))



svc()
#####
#####


#####
#####
# Cross validation
#####
#####

tuned_parameters_svc = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.1, 1, 10, 100, 1000]}]


tuned_parameters_svc = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.1, 1, 10, 100, 1000]}]


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


