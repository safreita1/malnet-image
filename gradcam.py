"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2020/05/14
Description: How to obtain a class activation heatmap for an image classification model.
"""

"""
Modified for use in MalNet by safreita1 
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow import keras
from collections import defaultdict

from process import create_image_symlinks
from main import get_generators, get_loss, evaluate


def get_img_array(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0) / 255.

    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(np.swapaxes(img_array, 0, -1))
        tape.watch(last_conv_layer_output)

        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def run_gradcam2(file_path, model):
    img_array = get_img_array(file_path)

    last_conv_layer_name = 'stage4_unit2_conv2'
    classifier_layer_names = ['bn1', 'relu1', 'pool1', 'fc1', 'softmax']

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)

    # We load the original image
    img = keras.preprocessing.image.load_img(file_path)
    img = keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    save_path = os.getcwd() + '/gradcam/' + file_path.split('test/')[1]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    superimposed_img.save(save_path)


def create_visuals(model_path):
    from config import args

    args['malnet_tiny'] = False
    create_image_symlinks(args)
    train_gen, val_gen, test_gen = get_generators(args)

    args['y_train'] = train_gen.labels
    args['class_indexes'] = list(val_gen.class_indices.values())
    args['class_labels'] = list(val_gen.class_indices.keys())
    args['num_classes'] = len(val_gen.class_indices.keys())

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(d) for d in args['devices'])

    model = tf.keras.models.load_model(model_path, compile=False)

    loss, _ = get_loss(args)
    model.compile(loss=loss, optimizer='adam', metrics=[])

    y_pred, y_scores = evaluate(test_gen, model)
    y_true = test_gen.labels.tolist()

    images_to_view = []
    counts = defaultdict(int)

    for idx, label in enumerate(y_true):
        if counts[label] < 20:
            images_to_view.append(idx)
            counts[label] += 1

    for idx in images_to_view:
        if y_pred[idx] == y_true[idx]:
            path = test_gen.filepaths[idx]
            run_gradcam2(path, model)


def main():
    model_path = '/raid/sfreitas3/malnet-image/info/logs/num_excluded=0/group=type/color=grayscale/pretrain=False/model=resnet18_loss=categorical_crossentropy_reweight=effective_num_beta=0.999/epochs=100/best_model.pt'
    create_visuals(model_path)


if __name__ == '__main__':
    main()

