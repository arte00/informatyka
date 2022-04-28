from platform import node
import sys
from tkinter import image_names
from cv2 import bitwise_and
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy as sp
from tqdm import tqdm

def load_image(path, infilename) :
    img = Image.open(path + infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj,np.ndarray):
        size=obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def encode_rle(_data):

    data_flatten = _data.copy()
    shape = data_flatten.shape
    dimension = len(shape)

    data_flatten = data_flatten.flatten()
    
    # structure = [dimension, shape[0], shape[1] + ... + shape[x] + (repeats + bit) + (repeats + bit) + ...]
    # worst case = 2 times more bits than in original data + dimensions + shape[0] + shape[1] + ... + shape[x]
    encoded_data = np.zeros(data_flatten.shape[0]*2 + dimension + 1).astype(_data.dtype)
    encoded_data[0] = dimension

    for counter, sh in enumerate(shape):
        encoded_data[counter + 1] = sh

    # pair = bit and how many times bit occured in row [1, 1, 4, 5, 5, 5] = [2, 1, 1, 4, 3, 5]
    pair_index = dimension + 1
    bit_index = 0
    with tqdm(total=data_flatten.shape[0]) as pbar:
        while bit_index < data_flatten.shape[0]:
            current_bit = data_flatten[bit_index]
            repeats = 1
            while bit_index + repeats < data_flatten.shape[0] and current_bit == data_flatten[bit_index + repeats]:
                repeats += 1

            bit_index += repeats
            pbar.update(bit_index)
            encoded_data[pair_index] = repeats
            encoded_data[pair_index + 1] = current_bit
            pair_index += 2

    return encoded_data[:pair_index]
    

def decore_rle(_data):
    
    data = _data.copy()

    dimension = data[0]
    shape = np.zeros(dimension).astype(np.int)

    for i in range(dimension):
        shape[i] = data[i + 1]

    size = 1
    for s in shape:
        size *= s

    decoded_data = np.zeros(size).astype(_data.dtype)

    decoded_index = 0
    for i in range(dimension + 1, data.shape[0], 2):
        repeats = data[i]
        bit = data[i+1]
        for j in range(repeats):
            decoded_data[decoded_index + j] = bit
        decoded_index += repeats

    decoded_data = np.reshape(decoded_data, shape).astype(data.dtype)

    return decoded_data

'''QUAD TREE'''

def split(data, level):
    
    color = np.mean(data, axis=(0, 1)).astype(int)

    quad_top_left = None
    quad_top_right = None
    quad_bot_right = None
    quad_bot_left = None

    if not (data == color).all() and data.shape[0] > 1 and data.shape[1] > 1:

        split_width = np.array_split(data, 2, axis=1)

        top_left, bot_left = np.array_split(split_width[0], 2, axis=0)
        top_right, bot_right = np.array_split(split_width[1], 2, axis=0)

        quad_top_left = split(top_left, level + 1)
        quad_top_right = split(top_right, level + 1)
        quad_bot_right = split(bot_right, level + 1)
        quad_bot_left = split(bot_left, level + 1)

    return (color, level, quad_top_left, quad_top_right, quad_bot_right, quad_bot_left, data.shape)

def join(top_left, top_right, bot_right, bot_left):
    top = np.concatenate(top_left, top_right, axis=1)
    bot = np.concatenate(bot_right, bot_left, axis=1)
    return np.concatenate(top, bot, axis=0)

def rle_testing():
    path = 'Lab06/'
    image = load_image(path, 'scan.png')
    size = get_size(image)
    
    encoded = encode_rle(image)
    decoded = decore_rle(encoded)

    img_size = get_size(image)
    enc_size = get_size(encoded)
    dec_size = get_size(decoded)

    print("image size", img_size)
    print("compressed image size", enc_size)
    print("decompressed image size", dec_size)
    print("compression ratio", round(dec_size/enc_size*100, 2), "%")
    
    plt.imshow(decoded, cmap='gray', vmin=0, vmax=255)
    plt.show()

def quad_split(data):

    level = 0

    color = np.mean(data, axis=(0, 1)).astype(int)
    split_width = np.array_split(data, 2, axis=1)

    top_left, bot_left = np.array_split(split_width[0], 2, axis=0)
    top_right, bot_right = np.array_split(split_width[1], 2, axis=0)

    if not (data == color).all() and data.shape[0] > 1 and data.shape[1] > 1:

        split_width = np.array_split(data, 2, axis=1)

        top_left, bot_left = np.array_split(split_width[0], 2, axis=0)
        top_right, bot_right = np.array_split(split_width[1], 2, axis=0)

        quad_top_left = split(top_left, level + 1)
        quad_top_right = split(top_right, level + 1)
        quad_bot_right = split(bot_right, level + 1)
        quad_bot_left = split(bot_left, level + 1)

    return (color, level, quad_top_left, quad_top_right, quad_bot_right, quad_bot_left, data.shape)

color = 0
level = 1
top_left = 2
top_right = 3
bot_right = 4
bot_left = 5
shape = 6

def join(top_left, top_right, bot_right, bot_left):

    top = np.concatenate((top_left, top_right), axis=1)
    bot = np.concatenate((bot_left, bot_right), axis=1)

    return np.concatenate((top, bot))

def quad_join(tree, max_level=-1):

    is_leaf = tree[top_left] is None and tree[top_right] is None and tree[bot_right] is None and tree[bot_left] is None

    if tree is None:
        return None

    if is_leaf or tree[level] == max_level:
        return np.tile(tree[color], (tree[shape][0], tree[shape][1], 1))
    else:
        return  join(
                quad_join(tree[top_left], max_level),
                quad_join(tree[top_right], max_level),
                quad_join(tree[bot_right], max_level),
                quad_join(tree[bot_left], max_level)).astype(int)

def search_leaves(tree, leaves):

    if tree is None:
        return

    if tree[top_left] == None and tree[top_right] == None and tree[bot_right] == None and tree[bot_left] == None:
        leaves.append(tree)
    
    search_leaves(tree[top_left], leaves)
    search_leaves(tree[top_right], leaves)
    search_leaves(tree[bot_right], leaves)
    search_leaves(tree[bot_left], leaves)



if __name__ == '__main__':

    # path = 'Lab06/'
    # title = 'rysunek.png'
    # image = load_image(path, title)
    # size = get_size(image)
    
    # encoded = encode_rle(image)
    # decoded = decore_rle(encoded)

    # img_size = get_size(image)
    # enc_size = get_size(encoded)
    # dec_size = get_size(decoded)

    # print
    # print("image size               ", img_size)
    # print("compressed image size    ", enc_size)
    # print("decompressed image size  " , dec_size)
    # print("compression ratio        ", round(img_size/enc_size, 3))
    # print("compressed to original   ", round(enc_size/img_size*100, 3), "%")
    # print("are equal                ", (image==decoded).all())
    
    # plt.imshow(decoded, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    path = 'Lab06/'
    data = load_image(path, 'milky.jpg')

    encoded = quad_split(data)
    decoded = quad_join(encoded, -1)

    img_size = get_size(data)
    enc_size = get_size(encoded)
    dec_size = get_size(decoded)

    print("QUAD------------------")
    print("original size        ", get_size(data))
    print("compressed size      ", get_size(encoded))
    print("decompressed size    ", get_size(decoded))
    print("compression ratio        ", round(img_size/enc_size, 3))
    print("compressed to original   ", round(enc_size/img_size*100, 3), "%")

    plt.imshow(data)
    plt.imshow(decoded)
    plt.show()


