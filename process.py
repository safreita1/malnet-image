import os


def filter_types(args, lines):
    types_to_exclude = ['adware', 'trojan', 'benign', 'riskware']

    filtered_files = []
    files = [args['image_dir'] + file.strip() + '.png' for file in lines]

    for file in files:
        mtype = file.split('malnet-images/')[1].rsplit('/', 2)[0]
        if mtype not in types_to_exclude:
            filtered_files.append(file)

    return filtered_files


def get_split_info(args):
    with open(os.getcwd() + '/split_info/{}/train.txt'.format(args['group']), 'r') as f:
        lines_train = f.readlines()

    with open(os.getcwd() + '/split_info/{}/val.txt'.format(args['group']), 'r') as f:
        lines_val = f.readlines()

    with open(os.getcwd() + '/split_info/{}/test.txt'.format(args['group']), 'r') as f:
        lines_test = f.readlines()

    if args['malnet_tiny']:
        files_train = filter_types(args, lines_train)
        files_val = filter_types(args, lines_val)
        files_test = filter_types(args, lines_test)
    else:
        files_train = [args['image_dir'] + file.strip() + '.png' for file in lines_train]
        files_val = [args['image_dir'] + file.strip() + '.png' for file in lines_val]
        files_test = [args['image_dir'] + file.strip() + '.png' for file in lines_test]

    if args['group'] == 'type':
        labels = sorted(list(set([file.split('malnet-images/')[1].rsplit('/', 2)[0] for file in files_train])))
        label_dict = {t: idx for idx, t in enumerate(labels)}

        train_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[0]] for file in files_train]
        val_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[0]] for file in files_val]
        test_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[0]] for file in files_test]

    elif args['group'] == 'family':
        labels = sorted(list(set([file.split('malnet-images/')[1].rsplit('/', 2)[1] for file in files_train])))
        label_dict = {t: idx for idx, t in enumerate(labels)}

        train_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[1]] for file in files_train]
        val_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[1]] for file in files_val]
        test_labels = [label_dict[file.split('malnet-images/')[1].rsplit('/', 2)[1]] for file in files_test]

    elif args['group'] == 'binary':
        labels = ['benign', 'malicious']
        label_dict = {t: idx for idx, t in enumerate(labels)}

        train_labels = [0 if 'benign' in file.split('malnet-images/')[1].rsplit('/', 2)[0] else 1 for file in files_train]
        val_labels = [0 if 'benign' in file.split('malnet-images/')[1].rsplit('/', 2)[0] else 1 for file in files_val]
        test_labels = [0 if 'benign' in file.split('malnet-images/')[1].rsplit('/', 2)[0] else 1 for file in files_test]

    else:
        print('Group does not exist')
        exit(1)

    print('Number of train samples: {}, val samples: {}, test samples: {}'.format(len(files_train), len(files_val), len(files_test)))

    return files_train, files_val, files_test, train_labels, val_labels, test_labels, label_dict


def create_image_symlinks(args):
    print('Creating image symlinks')

    files_train, files_val, files_test, _, _, _, _ = get_split_info(args)

    # create symlinks for train/val/test folders
    dst_dir = args['data_dir'] + 'malnet_tiny={}/{}/'.format(args['malnet_tiny'], args['group'])

    for src_path in files_train:
        dst_path = src_path.replace(args['image_dir'], dst_dir + 'train/')

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
        dst_path = src_path.replace(args['image_dir'], dst_dir + 'val/')

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
        dst_path = src_path.replace(args['image_dir'], dst_dir + 'test/')

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
