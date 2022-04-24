import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    while bit_index < data_flatten.shape[0]:
        current_bit = data_flatten[bit_index]
        repeats = 1
        while bit_index + repeats < data_flatten.shape[0] and current_bit == data_flatten[bit_index + repeats]:
            repeats += 1

        bit_index += repeats
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


if __name__ == '__main__':
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