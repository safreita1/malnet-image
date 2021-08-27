import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy
import numpy as np
import multiprocessing
import tensorflow as tf

from tqdm import tqdm
from pprint import pprint
from tensorflow import keras
from joblib import Parallel, delayed
from classification_models.keras import Classifiers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score

from losses import get_loss
from process import create_image_symlinks

models = {
    'resnet50': keras.applications.ResNet50,
    'resnet101': keras.applications.ResNet101,
    'densenet121': keras.applications.DenseNet121,
    'densenet169': keras.applications.DenseNet169,
    'mobilenet': keras.applications.MobileNet,
    'mobilenetv2': keras.applications.MobileNetV2
}


class Metrics(Callback):
    def __init__(self, args, val_gen):
        super(Metrics, self).__init__()
        self.args = args
        self.val_gen = val_gen
        self.best_macro_f1 = 0

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            y_pred, y_scores = evaluate(self.val_gen, self.model)
            y_true = self.val_gen.classes.tolist()

            macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
            tf.summary.scalar('macro-f1', data=macro_f1, step=epoch)

            if macro_f1 > self.best_macro_f1 or epoch == 0:
                print('Improved macro-F1 from {} to {} at epoch {}. Saving and logging model.'.format(self.best_macro_f1, macro_f1, epoch))
                self.best_macro_f1 = macro_f1

                self.save_model()
                log_info(self.args, epoch, y_true, y_pred, y_scores, data_type='val')

    def save_model(self):
        self.model.save(self.args['log_dir'] + 'best_model.pt')


def log_info(args, epoch, y_true, y_pred, y_scores, data_type='val'):
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
    report = classification_report(y_true, y_pred, labels=args['class_indexes'], target_names=args['class_labels'], output_dict=True)

    with open(args['log_dir'] + 'best_{}_info.txt'.format(data_type), 'w') as f:
        pprint('Parameters', stream=f)
        pprint(args, stream=f)

        pprint('Classification report', stream=f)
        pprint(report, stream=f)
        pprint('Best model at epoch: {}'.format(epoch), stream=f)
        pprint('Macro-f1: {}'.format(macro_f1), stream=f)
        pprint('Malware group: {}'.format(args['group']), stream=f)

        pprint('Confusion matrix', stream=f)
        cm = confusion_matrix(y_true, y_pred, labels=args['class_indexes'])
        pprint(cm, stream=f)
        pprint('Label dictionary:', stream=f)
        pprint(args['class_labels'], stream=f)

        if args['group'] == 'binary':
            pprint('FPR/TPR Info', stream=f)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=[1])
            pprint('tpr: {}'.format(tpr.tolist()), stream=f)
            pprint('fpr: {}'.format(fpr.tolist()), stream=f)
            pprint('thresholds: {}'.format(thresholds.tolist()), stream=f)

            auc_macro_score = roc_auc_score(y_true, y_scores, average='macro')
            auc_class_scores = roc_auc_score(y_true, y_scores, average=None)
            pprint('AUC macro score: {}'.format(auc_macro_score), stream=f)
            pprint('AUC class scores: {}'.format(auc_class_scores), stream=f)


def evaluate(data_gen, model):
    y_pred = []
    y_scores = []

    batches = 0
    for x, y in tqdm(data_gen):
        pred = model.predict(x)

        y_pred.extend(np.argmax(pred, axis=1).tolist())
        y_scores.extend(pred[:, 1].tolist())  # only used in binary setting

        batches += 1
        if batches == len(data_gen):
            break  # we need to break the loop by hand because the generator loops indefinitely

    return y_pred, y_scores


def build_transfer_model(args, base_model):
    base_model.trainable = True

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(args['num_classes'], activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model


def train_model(args, train_gen, val_gen):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(d) for d in args['devices'])

    args['log_dir'] = os.getcwd() + '/info/logs/malnet_tiny={}/group={}/color={}/pretrain={}/model={}_loss={}_alpha={}_reweight={}_beta={}/epochs={}/'.format(
        args['malnet_tiny'], args['group'], args['color_mode'], args['weights'], args['model'], args['loss'], args['alpha'], args['reweight'], args['reweight_beta'],  args['epochs'])
    os.makedirs(args['log_dir'], exist_ok=True)

    if args['color_mode'] == 'grayscale':
        args['num_channels'] = 1
    else:
        args['num_channels'] = 3

    input_shape = (args['im_size'], args['im_size'], args['num_channels'])

    if args['weights'] is 'imagenet':
        model = models[args['model']](include_top=False, weights=args['weights'], input_shape=input_shape, classes=args['num_classes'])
        model = build_transfer_model(args, model)
    else:
        if 'mobile' in args['model']:
            model = models[args['model']](weights=args['weights'], input_shape=input_shape, classes=args['num_classes'], alpha=args['alpha'])
        elif 'resnet18' in args['model']:
            args['alpha'] = 0
            model, _ = Classifiers.get(args['model'])
            model = model(weights=args['weights'], input_shape=input_shape, classes=args['num_classes'])
        else:
            args['alpha'] = 0
            model = models[args['model']](weights=args['weights'], input_shape=input_shape, classes=args['num_classes'])

    loss, class_weights = get_loss(args)
    model.compile(loss=loss, optimizer='adam', metrics=[])

    model.fit(
        train_gen,
        batch_size=args['batch_size'],
        steps_per_epoch=int(train_gen.samples / args['batch_size']),
        epochs=args['epochs'],
        class_weight=class_weights,
        callbacks=[Metrics(args, val_gen)],
        workers=multiprocessing.cpu_count(),
    )

    # load best model
    model = tf.keras.models.load_model(args['log_dir'] + 'best_model.pt', compile=False)
    model.compile(loss=loss, optimizer='adam', metrics=[])

    return model


def get_generators(args):
    train_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}malnet_tiny={}/{}/train'.format(args['data_dir'], args['malnet_tiny'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=True
    )

    val_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}malnet_tiny={}/{}/val'.format(args['data_dir'], args['malnet_tiny'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=False
    )

    test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}malnet_tiny={}/{}/test'.format(args['data_dir'], args['malnet_tiny'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def run(args_og, group, device):
    args = copy.deepcopy(args_og)

    args['devices'] = [device]
    args['group'] = group

    create_image_symlinks(args)
    train_gen, val_gen, test_gen = get_generators(args)

    args['y_train'] = train_gen.labels
    args['class_indexes'] = list(val_gen.class_indices.values())
    args['class_labels'] = list(val_gen.class_indices.keys())
    args['num_classes'] = len(val_gen.class_indices.keys())

    model = train_model(args, train_gen, val_gen)

    y_pred, y_scores = evaluate(test_gen, model)
    y_true = test_gen.classes.tolist()

    log_info(args, 'based on best val model', y_true, y_pred, y_scores, data_type='test')


def model_experiments():
    from config import args

    devices = [1]  # [1, 2, 3]
    groups = ['type']  # , 'type', 'family'

    Parallel(n_jobs=len(groups))(
        delayed(run)(args, group, devices[idx])
        for idx, group in enumerate(tqdm(groups)))


if __name__ == '__main__':
    model_experiments()
