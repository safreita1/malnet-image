from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score

from main import get_data, evaluate


def get_cm():
    args = {
        'im_size': 256,
        'num_channels': 1,
        'group': 'type',
        'batch_size': 128,
        'color_mode': 'grayscale',  # rgb, grayscale

        'epochs': 100,
        'model': 'resnet18',
        'pretrain': False,
        'types_to_exclude': [],

        'seed': 1,
        'devices': [0],
        'gpu_list': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7'],
        'base_dir': '/localscratch/sfreitas3/',
        'data_dir': '/localscratch/sfreitas3/malnet-image/data',
        'image_dir': '/localscratch/sfreitas3/malnet-image-256x256/',
    }

    train_gen, val_gen, test_gen = get_data(args)
    class_indexes = list(train_gen.class_indices.values())
    class_labels = list(train_gen.class_indices.keys())

    label_idx, counts = np.unique(train_gen.labels.tolist() + val_gen.labels.tolist() + test_gen.labels.tolist(), return_counts=True)
    sorted_counts, sorted_labels, sorted_idx = zip(*sorted(zip(counts, class_labels, class_indexes), reverse=True))

    new_sorted_idx = defaultdict()
    for i, idx in enumerate(sorted_idx):
        new_sorted_idx[idx] = i

    if not os.path.exists('type_test_cm_new.csv'):
        model_path = '/raid/sfreitas3/malnet-image/info/logs/group=type/color=grayscale/pretrain=False/model=resnet18,epochs=100/best_model.pt'

        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', metrics=[])

        # test data
        y_pred, y_scores = evaluate(test_gen, model)
        y_true = test_gen.classes.tolist()

        y_pred_new = []
        for y in y_pred:
            y_new = new_sorted_idx[y]
            y_pred_new.append(y_new)

        y_true_new = []
        for y in y_true:
            y_new = new_sorted_idx[y]
            y_true_new.append(y_new)

        cm = confusion_matrix(y_true_new, y_pred_new)
        np.savetxt("type_test_cm_new.csv", cm, delimiter=",")

    cm_test = pd.read_csv('type_test_cm_new.csv', names=sorted_labels)
    return cm_test


def plot_heatmap(cm, data_type='test'):
    figsize = (30, 30)
    labels = list(cm.columns.values)
    cm = cm.to_numpy()

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    cm = cm.to_numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    cm_anno = cm.tolist()
    for o_idx, outer in enumerate(cm):
        for i_idx, element in enumerate(outer):
            if round(element, 2) == 0.00:
                cm_anno[o_idx][i_idx] = str(0)
            else:
                cm_anno[o_idx][i_idx] = str(element).lstrip('0')

    ax = sns.heatmap(cm, annot=pd.DataFrame(cm_anno), fmt='', ax=ax, cmap='Blues', square=True, xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='light', size=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light', size=16)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(data_type + '_cm.pdf')
    plt.show()
    plt.clf()


def create_heatmap():
    cm = get_cm()
    # plot_heatmap(cm_val, data_type='val')
    plot_heatmap(cm, data_type='test')

    # with open(type_path, 'r') as f:
    #     info = f.read()
    #     info = json.loads(info.split('Classification report')[1].split('Best model at epoch')[0].split('\'\n')[1].replace("\'", "\"").strip()[:-2])
    #     print('hi')


def create_roc_curves():


    # binary_path = '/raid/sfreitas3/malnet-image/info/logs/group=binary/color=grayscale/pretrain=False/model=resnet18,epochs=100/best_test_info.txt'
    #
    # with open(binary_path, 'r') as f:
    #     info = f.read()
    #     auc = info.split('AUC macro score: ')[1].split('AUC class')[0].replace('\n', '').replace("'", '')
    #     tpr, fpr = info.split('FPR/TPR Info')[1].split('AUC macro score')[0].split('fpr: ')
    #     tpr = tpr.split('tpr: ')[1].replace(')', '').replace('(', '').replace("'", '')
    #     fpr, thresholds = fpr.split('thresholds: ')
    #     fpr = fpr.replace("'", '').replace(')', '').replace('(', '')
    #     thresholds = thresholds.replace("'", '').replace(')', '')
    #     tpr, fpr, thresholds = eval(tpr), eval(fpr), eval(thresholds)

    cm_test, auc_macro_score, tpr, fpr, thresholds = get_cm()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="auc=" + str(round(float(auc_macro_score), 4)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=4)
    plt.title('Positive label = 0')
    plt.savefig('roc_curve_0.pdf')
    plt.show()
    plt.clf()


def get_fp_tp_values():
    binary_path = '/raid/sfreitas3/malnet-image/info/logs/group=binary/color=grayscale/pretrain=False/model=resnet18,epochs=100/best_test_info.txt'

    with open(binary_path, 'r') as f:
        info = f.read()
        tpr, fpr = info.split('FPR/TPR Info')[1].split('AUC macro score')[0].split('fpr: ')
        tpr = tpr.split('tpr: ')[1].replace(')', '').replace('(', '').replace("'", '')
        fpr, thresholds = fpr.split('thresholds: ')
        fpr = fpr.replace("'", '').replace(')', '').replace('(', '')
        thresholds = thresholds.replace("'", '').replace(')', '')

        tpr, fpr, thresholds = eval(tpr), eval(fpr), eval(thresholds)
        print('hi')


if __name__ == '__main__':
    create_heatmap()
    # create_roc_curves()
    # get_fp_tp_values()