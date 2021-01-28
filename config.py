import os

args = {
    'im_size': 256,
    'batch_size': 128,
    'num_channels': 1,
    'malnet_tiny': False,
    'group': 'type',  # options: binary, type, family
    'color_mode': 'grayscale',  # options: rgb, grayscale

    'epochs': 100,
    'model': 'resnet18',
    'alpha': 1.0,  # sets MobileNet model size
    'weights': None,  # options: None (which is random), imagenet
    'loss': 'categorical_crossentropy',  # options: categorical_crossentropy, categorical_focal_loss
    'reweight': None,  # options: None, effective_num
    'reweight_beta': 0.999,  # used if 'reweight' is set to effective_num

    'seed': 1,
    'devices': [1],
    'data_dir': os.getcwd() + '/data/',  # symbolic link directory
    'image_dir': '/raid/sfreitas3/malnet-images/',  # path where data is located

}
