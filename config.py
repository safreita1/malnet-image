args = {
    'im_size': 256,
    'num_channels': 1,
    'group': 'family',
    'batch_size': 128,
    'color_mode': 'grayscale',  # rgb, grayscale

    'epochs': 100,
    'model': 'resnet18',
    'pretrain': False,

    'seed': 1,
    'devices': [0],
    'gpu_list': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7'],
    'base_dir': '/localscratch/sfreitas3/',
    'data_dir': '/localscratch/sfreitas3/malnet-image/data',
    'image_dir': '/localscratch/sfreitas3/malnet-image-256x256/',
    'loss_type': 'categorical_focal_loss',  # categorical_crossentropy, categorical_focal_loss
    'reweight_method': 'effective_num',  # effective_num, None
    'reweight_beta': 0.999  # used if 'reweight_method' is set to effective_num
}


if args['color_mode'] == 'grayscale':
    args['num_channels'] = 1
