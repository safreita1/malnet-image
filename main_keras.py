import os
import copy
import itertools
import multiprocessing
import imgaug.augmenters as iaa
from joblib import Parallel, delayed

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tqdm import tqdm
from pprint import pprint
from tensorflow import keras
from classification_models.keras import Classifiers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score

from process import create_image_symlinks
from losses import categorical_focal_loss


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


# https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast
#         preprocessing_function=add_gaussian_noise,
def add_gaussian_noise(img):
    noisy_img = iaa.AdditiveGaussianNoise(scale=0.4*255).augment_image(img)
    np.clip(noisy_img, 0., 255.)
    return noisy_img


def add_poisson_noise(img):
    noisy_img = iaa.AdditivePoissonNoise(lam=5.0).augment_image(img)
    np.clip(noisy_img, 0., 255.)
    return noisy_img

def get_effective_class_weights(args):
    '''
        Determines class weighting according to the following paper
        - https://arxiv.org/abs/1901.05555
    ''' 

    unique, class_frequencies = np.unique(args['y_train'], return_counts=True)
    effective_num = [(1-args['reweight_beta']) / (1 - np.power(args['reweight_beta'], c_i)) for c_i in class_frequencies]
    class_weights = effective_num / sum(effective_num) * args['num_classes']
    print('calculated class frequencies')
    class_weights = {k:v for k,v in enumerate(class_weights)}
    return class_weights


def evaluate(data_gen, model):
    y_pred = []
    y_scores = []

    batches = 0
    for x, y in tqdm(data_gen):
        pred = model.predict(x)

        y_pred.extend(np.argmax(pred, axis=1).tolist())
        y_scores.extend(pred[:, 1].tolist())

        batches += 1
        if batches == len(data_gen):
            break  # we need to break the loop by hand because the generator loops indefinitely

    return y_pred, y_scores


def build_transfer_model(args, base_model):
    base_model.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(args['num_classes'], activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model


def train_model(args, train_gen, val_gen):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(d) for d in args['devices'])

    args['log_dir'] = os.getcwd() + '/info/logs/group={}/color={}/pretrain={}/model={}_loss={}_reweight={}_beta={}/epochs={}/'.format(args['group'], args['color_mode'],
                                                                                                                            args['pretrain'], args['model'], args['loss_type'],
                                                                                                                            args['reweight_method'], args['reweight_beta'],  args['epochs'])
    os.makedirs(args['log_dir'], exist_ok=True)

    input_shape = (args['im_size'], args['im_size'], args['num_channels'])

    if args['pretrain']:
        ModelType, _ = Classifiers.get(args['model'])
        model = ModelType(include_top=False, weights='imagenet', classes=args['num_classes'], input_shape=input_shape)
        model = build_transfer_model(args, model)
    else:
        ModelType, _ = Classifiers.get(args['model'])
        model = ModelType(include_top=True, weights=None, classes=args['num_classes'], input_shape=input_shape)

    if args['reweight_method'] == 'effective_num':
        class_weights = get_effective_class_weights(args)  
    else:
        class_weights = [1.0/args['num_classes']] * args['num_classes']

    if args['loss_type'] == 'categorical_focal_loss':
        if type(class_weights) is dict:
            alpha = [class_weights[i] for i in class_weights.keys()]
        else:
            alpha = class_weights
        loss = [categorical_focal_loss(alpha=[alpha], gamma=2)]
        class_weights = {i: 1.0/args['num_classes'] for i in range(args['num_classes'])}  # class weighting already incorporated in focal loss alpha
    else:
        loss = args['loss_type']

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


def get_data(args):
    train_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}/{}/train'.format(args['data_dir'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=True
    )

    val_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}/{}/val'.format(args['data_dir'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=False
    )

    test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory='{}/{}/test'.format(args['data_dir'], args['group']),
        class_mode='categorical',
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def model_experiments():
    from config import args

    models = ['resnet18']
    groups = ['binary', 'family', 'type']
    pretraining = [False]

    loss_type = 'categorical_focal_loss'  # categorical_crossentropy, categorical_focal_loss
    reweight_method = 'effective_num'  # effective_num, None

    params = list(itertools.product(*[models, groups, pretraining]))

    Parallel(n_jobs=1)(
        delayed(run)(args, idx, model, group, pretrain, loss_type, reweight_method)
        for idx, (model, group, pretrain) in enumerate(tqdm(params)))


def run(args_og, idx, model, group, pretrain, loss_type, reweight_method):
    # idx += 3
    args = copy.deepcopy(args_og)
    args['devices'] = [idx]
    args['model'] = model
    args['group'] = group
    args['pretrain'] = pretrain
    args['loss_type'] = loss_type
    args['reweight_method'] = reweight_method

    y_train, y_val, y_test = create_image_symlinks(args)
    train_gen, val_gen, test_gen = get_data(args)

    args['class_indexes'] = list(val_gen.class_indices.values())
    args['class_labels'] = list(val_gen.class_indices.keys())
    args['num_classes'] = len(val_gen.class_indices.keys())
    args['y_train'] = y_train

    model = train_model(args, train_gen, val_gen)

    y_pred, y_scores = evaluate(test_gen, model)
    y_true = test_gen.classes.tolist()

    log_info(args, 'based on best val model', y_true, y_pred, y_scores, data_type='test')


def main():
    from config import args

    y_train, y_val, y_test = create_image_symlinks(args)
    train_gen, val_gen, test_gen = get_data(args)

    args['class_indexes'] = list(val_gen.class_indices.values())
    args['class_labels'] = list(val_gen.class_indices.keys())
    args['num_classes'] = len(val_gen.class_indices.keys())
    args['y_train'] = y_train

    model = train_model(args, train_gen, val_gen)

    y_pred, y_scores = evaluate(test_gen, model)
    y_true = test_gen.classes.tolist()

    log_info(args, 'based on best val model', y_true, y_pred, y_scores, data_type='test')


if __name__ == '__main__':
    model_experiments()
    # main()