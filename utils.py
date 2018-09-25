import datetime
import itertools
import logging as log

import matplotlib
from sklearn import metrics

matplotlib.use("Agg")
import numpy as np
import yaml
from matplotlib import pyplot as plt

import pickle
import time
import json


def yaml_loader(filepath):
    with open(filepath, 'r') as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


def yaml_dump(filepath, data):
    with open(filepath, 'w') as file_descriptor:
        yaml.dump(data, file_descriptor)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + "_" + start_time + ".png")
    plt.clf()


def log_metrics_and_params(results, model_savepath):
    # log results and save path
    to_write = {}
    to_write['results'] = results
    to_write['model_savepath'] = model_savepath
    log.info('%s', json.dumps(to_write))


def save_model(clf):
    # save model with timestamp
    timestring = "".join(str(time.time()).split("."))
    model_savepath = 'model_' + timestring + '.pk'
    with open(model_savepath, 'wb') as ofile:
        pickle.dump(clf, ofile)
    return model_savepath


def get_train_metrics():
    # currently impossible
    # X_train and y_train are in higher scopes
    pass


def get_val_metrics(y_pred, y_true):
    return get_metrics(y_pred, y_true)


def get_metrics(y_pred, y_true):
    # compute more than just one metrics

    chosen_metrics = {
        # 'conf_mat': metrics.confusion_matrix,
        'accuracy': metrics.accuracy_score,
        'auc': metrics.roc_auc_score,
    }

    results = {}
    for metric_name, metric_func in chosen_metrics.items():
        try:
            inter_res = metric_func(y_pred, y_true)
        except Exception as ex:
            inter_res = None
            log.error("Couldn't evaluate %s because of %s", metric_name, ex)
        results[metric_name] = inter_res

    # results['conf_mat'] = results['conf_mat'].tolist()

    return results


def _my_scorer(clf, X_val, y_true_val):
    # do all the work and return some of the metrics

    y_pred_val = clf.predict(X_val)

    results = get_val_metrics(y_pred_val, y_true_val)
    timestring = "".join(str(time.time()).split("."))
    model_savepath = 'model_' + timestring + '.pk'
    log_metrics_and_params(results, model_savepath)
    return results['accuracy']


def plot_learning_curves(history, file_name):
    # summarize history for accuracy
    now = datetime.datetime.now()
    start_time = now.strftime("%Y-%m-%d %H:%M")
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    path = "reports/learning_curves_" + file_name + str(start_time) + ".png"
    plt.savefig(path)
    return path
