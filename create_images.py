import os
import binascii
import numpy as np
from PIL import Image
from collections import defaultdict
from androguard.core.bytecodes.dvm import DalvikVMFormat


def get_width(size_kb):
    if size_kb < 10:
        width = 32
    elif 10 <= size_kb < 30:
        width = 64
    elif 30 <= size_kb < 60:
        width = 128
    elif 60 <= size_kb < 100:
        width = 256
    elif 100 <= size_kb < 200:
        width = 384
    elif 200 <= size_kb < 500:
        width = 512
    elif 500 <= size_kb < 1000:
        width = 768
    else:
        width = 1024

    return width


def load_1d_dex(dex_file, max_size=80):
    # Convert a binary file to a 1d array by reading byte values.
    file_size = os.stat(dex_file).st_size >> 20  # in MB
    if file_size < max_size:

        # Read the binary file in hex
        with open(dex_file, 'rb') as fp:
            hexstring = binascii.hexlify(fp.read())

            # Convert hex string to byte array
            # 2 hex numbers give up to 256 (1 byte) in decimal
            byte_list = [int(hexstring[i: i + 2], 16) for i in
                         range(0, len(hexstring), 2)]

            return np.array(byte_list, dtype=np.uint8)


def get_dex_info(dex_file):
    dex_info = defaultdict(int)

    with open(dex_file, 'rb') as f:
        dex = DalvikVMFormat(f.read())
        dex_info['file_size'] = dex.get_header_item().file_size

        # DEX HEADER
        dex_info['header_size'] = dex.get_header_item().header_size

        # INDEX
        dex_info['string_ids_size'] = dex.get_header_item().string_ids_size
        dex_info['string_ids_off'] = dex.get_header_item().string_ids_off

        dex_info['type_ids_size'] = dex.get_header_item().type_ids_size
        dex_info['type_ids_off'] = dex.get_header_item().type_ids_off

        dex_info['proto_ids_size'] = dex.get_header_item().proto_ids_size
        dex_info['proto_ids_off'] = dex.get_header_item().proto_ids_off

        dex_info['field_ids_size'] = dex.get_header_item().field_ids_size
        dex_info['field_ids_off'] = dex.get_header_item().field_ids_off

        dex_info['method_ids_size'] = dex.get_header_item().method_ids_size
        dex_info['method_ids_off'] = dex.get_header_item().method_ids_off

        dex_info['class_defs_size'] = dex.get_header_item().class_defs_size
        dex_info['class_defs_off'] = dex.get_header_item().class_defs_off

        # DATA
        dex_info['data_size'] = dex.get_header_item().data_size
        dex_info['data_off'] = dex.get_header_item().data_off

        # LINK DATA
        dex_info['link_size'] = dex.get_header_item().link_size
        dex_info['link_off'] = dex.get_header_item().link_off

    return dex_info


def set_color(dex_info, dex_b, dex_r, dex_g):
    # HEADER -> red
    s0_end = dex_info['header_size']
    dex_r[0: s0_end] = dex_b[0:s0_end]
    dex_b[0: s0_end] = 0

    # DATA -> green
    s7_idx = dex_info['data_off']
    e7_idx = s7_idx + dex_info['data_size']
    dex_g[s7_idx:e7_idx] = dex_b[s7_idx:e7_idx]
    dex_b[s7_idx:e7_idx] = 0

    return dex_r, dex_g


def get_linear(dex_array, size_kb):
    width = get_width(size_kb)
    height = int(len(dex_array) / width)

    dex_array = dex_array[:height * width]  # trim array to 2-D shape
    dex_array_2d = np.reshape(dex_array, (height, width))

    return dex_array_2d


def create_linear_image(dex_path):
    dex_info = get_dex_info(dex_path)

    dex_b = load_1d_dex(dex_path)
    dex_r = np.zeros(len(dex_b), dtype=np.uint8)
    dex_g = np.zeros(len(dex_b), dtype=np.uint8)
    dex_r, dex_g = set_color(dex_info, dex_b, dex_r, dex_g)

    file_kb = int(os.path.getsize(dex_path) / 1000)
    b_c = get_linear(dex_b, file_kb)
    r_c = get_linear(dex_r, file_kb)
    g_c = get_linear(dex_g, file_kb)

    im = np.stack((r_c, g_c, b_c), axis=-1)
    im = Image.fromarray(im)

    image_sizes = [512, 256, 128, 64, 32]
    for size in image_sizes:
        image_path = dex_path.rsplit('/', 1)[0].replace('temp_data/', 'image_data_benign/{}/'.format(size)) + '.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        im_resized = im.resize(size=(size, size), resample=Image.ANTIALIAS)  # ANTIALIAS provides higher quality image
        im_resized.save(image_path)




