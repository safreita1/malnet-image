import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_rand(labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=1, shuffle=True):
    train_idx, test_idx = train_test_split(np.arange(len(labels)), train_size=train_ratio * 0.7, test_size=val_ratio + test_ratio, random_state=seed, shuffle=shuffle, stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, train_size=val_ratio/(val_ratio+test_ratio), test_size=test_ratio/(val_ratio+test_ratio), random_state=seed, shuffle=shuffle, stratify=np.asarray(labels)[test_idx])

    print('Number of train samples: {}, val samples: {}, test samples: {}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return train_idx, val_idx, test_idx


def get_data(args, limit=np.inf):
    files = []
    labels = []
    label_idx = 0
    label_dict = {}

    print('Loading image info...')
    if args['group'] == 'type':
        mtype_dirs = glob(args['image_dir'] + '*/')
        mtype_dirs = [mtype_dir for mtype_dir in mtype_dirs if '*' not in mtype_dir]
        m_paths = defaultdict(list)

        for mtype_dir in mtype_dirs:
            mtype = mtype_dir.split('/malnet-image-256x256/')[1].split('/')[0]
            m_paths[mtype] = glob(mtype_dir + '/*/*.png')

    elif args['group'] == 'family':
        mfam_dirs = glob(args['image_dir'] + '*/*/')
        mfam_dirs = [mtype_dir for mtype_dir in mfam_dirs if '*' not in mtype_dir]
        m_paths = defaultdict(list)

        for mfam_dir in mfam_dirs:
            mfam = mfam_dir.split('/malnet-image-256x256/')[1].split('/')[1]
            m_paths[mfam].extend(glob(mfam_dir + '/*.png'))

    elif args['group'] == 'binary':
        binary_dirs = glob(args['image_dir'] + '*/')
        bin_dirs = [bin_dir for bin_dir in binary_dirs if '*' not in bin_dir]
        m_paths = defaultdict(list)

        for bin_dir in bin_dirs:
            if 'benign' in bin_dir:
                m_paths['benign'] = glob(bin_dir + '/*/*.png')
            else:
                m_paths['malicious'].extend(glob(bin_dir + '/*/*.png'))
    else:
        print('Group not valid')
        exit(1)

    for mal_group, paths in m_paths.items():
        if len(paths) < limit:
            files.extend(paths)
            labels.extend([label_idx] * len(paths))
            label_dict[mal_group] = label_idx
            label_idx += 1

    files, labels = (list(t) for t in zip(*sorted(zip(files, labels))))

    print('Finished loading image info')
    return files, labels, label_dict


def create_image_symlinks(args):
    print('Creating image symlinks')

    files, labels, label_dict = get_data(args)
    train_idx, val_idx, test_idx = split_rand(labels)

    files_train, y_train = np.asarray(files)[train_idx].tolist(), np.asarray(labels)[train_idx]
    files_val, y_val = np.asarray(files)[val_idx].tolist(), np.asarray(labels)[val_idx]
    files_test, y_test = np.asarray(files)[test_idx].tolist(), np.asarray(labels)[test_idx]

    # create symlinks for train/val/test folders
    dst_dir = args['data_dir'] + '/{}/'.format(args['group'])

    for src_path in files_train:
        dst_path = src_path.replace('/localscratch/sfreitas3/malnet-image-256x256/', dst_dir + 'train/')

        if args['group'] == 'binary':
            if 'benign' not in dst_path:
                dst_path = dst_path.split('train/')[0] + 'train/malicious/' + dst_path.split('train/')[1].split('/')[2]
            else:
                dst_path = dst_path.split('train/')[0] + 'train/benign/' + dst_path.split('train/')[1].split('/')[2]

        elif args['group'] == 'family':
            dst_path = dst_path.split('train/')[0] + 'train/' + '/'.join(dst_path.split('train/')[1].split('/')[1:3])

        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.symlink(src_path, dst_path)

    for src_path in files_val:
        dst_path = src_path.replace('/localscratch/sfreitas3/malnet-image-256x256/', dst_dir + 'val/')

        if args['group'] == 'binary':
            if 'benign' not in dst_path:
                dst_path = dst_path.split('val/')[0] + 'val/malicious/' + dst_path.split('val/')[1].split('/')[2]
            else:
                dst_path = dst_path.split('val/')[0] + 'val/benign/' + dst_path.split('val/')[1].split('/')[2]

        elif args['group'] == 'family':
            dst_path = dst_path.split('val/')[0] + 'val/' + '/'.join(dst_path.split('val/')[1].split('/')[1:3])

        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.symlink(src_path, dst_path)

    for src_path in files_test:
        dst_path = src_path.replace('/localscratch/sfreitas3/malnet-image-256x256/', dst_dir + 'test/')

        if args['group'] == 'binary':
            if 'benign' not in dst_path:
                dst_path = dst_path.split('test/')[0] + 'test/malicious/' + dst_path.split('test/')[1].split('/')[2]
            else:
                dst_path = dst_path.split('test/')[0] + 'test/benign/' + dst_path.split('test/')[1].split('/')[2]

        elif args['group'] == 'family':
            dst_path = dst_path.split('test/')[0] + 'test/' + '/'.join(dst_path.split('test/')[1].split('/')[1:3])

        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.symlink(src_path, dst_path)

    print('Finished creating symlinks')