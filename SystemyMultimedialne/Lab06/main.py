from platform import node
import sys
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


def save_image(npdata, outfilename) :
    img = Image.fromarray(np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save(outfilename)

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

node_counter = 0

def encode_quad(data, max=-1, level=0):
    global node_counter
    node_counter += 1

    color = color = np.mean(data, axis=(0, 1)).astype(data.dtype)

    quad_top_left = None
    quad_top_right = None
    quad_bot_right = None
    quad_bot_left = None

    if not (data == color).all() and level != max and data.shape[0] > 1 and data.shape[1] > 1:
        split_width = np.array_split(data, 2, axis=1)
        top_left, bot_left = np.array_split(split_width[0], 2, axis=0)
        top_right, bot_right = np.array_split(split_width[1], 2, axis=0)
        quad_top_left = encode_quad(top_left, max, level + 1)
        quad_top_right = encode_quad(top_right, max, level + 1)
        quad_bot_right = encode_quad(bot_right, max, level + 1)
        quad_bot_left = encode_quad(bot_left, max, level + 1)

    return (color, level, quad_top_left, quad_top_right, quad_bot_right, quad_bot_left)

def decode_quad(node):
    pass

def join(top_left, top_right, bot_right, bot_left):
    top = np.concatenate(top_left, top_right, axis=1)
    bot = np.concatenate(bot_right, bot_left, axis=1)
    return np.concatenate(top, bot, axis=0)

def rle_testing():
    path = 'Lab06/'
    image = load_image(path, 'rysunek_techniczny.jpg')
    size = get_size(image)
    
    encoded = encode_rle(image)
    decoded = decore_rle(encoded)

    print(len(image))
    print(len(image.flatten()))
    print(len(encoded))
    print(len(decoded))
    
    plt.imshow(decoded, cmap='gray', vmin=0, vmax=255)
    plt.show()


def quad_testing():

    # image = np.dstack([np.eye(7),np.eye(7),np.eye(7)])
    path = 'Lab06/'
    # _image = load_image(path, 'monkey.jpg')
    _image = np.eye(8)
    image = _image.copy()
    print("image", get_size(image))
    print(image)
    splitted = encode_quad(image, 1)
    print(splitted[2])
    print("splitted", get_size(splitted))
    print("node_counter", node_counter)


if __name__ == '__main__':
    quad_testing()