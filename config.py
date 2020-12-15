args = {
    'im_size': 256,
    'num_channels': 3,
    'group': 'family',
    'batch_size': 128,
    'color_mode': 'rgb',  # rgb, grayscale

    'epochs': 100,
    'model': 'resnet18',
    'pretrain': True,

    'seed': 1,
    'devices': [6],
    'gpu_list': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7'],
    'base_dir': '/localscratch/sfreitas3/',
    'data_dir': '/localscratch/sfreitas3/malnet-image/data',
    'image_dir': '/localscratch/sfreitas3/malnet-image-256x256/'
}

args['batch_size'] = args['batch_size'] * len(args['devices'])

if args['color_mode'] == 'grayscale':
    args['num_channels'] = 1
