from re import L
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.fftpack
from tqdm import tqdm

class ImageInfo:

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Data:

    def __init__(self) -> None:
        self.y = None
        self.cb = None
        self.cr = None


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

def chroma_subsample(layer, chroma_mode):

    if chroma_mode == "4:4:4":
        return layer
    elif chroma_mode == "4:2:2":
        output = np.empty((layer.shape[0], int(layer.shape[1]/2)))
        for i in range(0, layer.shape[0]):
            for j in range(0, layer.shape[1], 2):
                output[i][int(j/2)] = layer[i][j]
        return output

def chroma_resample(_data, sampling):
    data = _data
    new_data = np.zeros((data.shape[0], data.shape[1]*2))

    if sampling == "4:4:4":
        new_data = data
    elif sampling == "4:2:2":
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_data[i, j*2] = data[i, j]
                new_data[i, j*2+1] = data[i, j]
    return new_data

def compress(layer, sampling, q, is_y):


    sampled = chroma_subsample(layer, sampling)
    # if not is_y:
    #     sampled = chroma_subsample(layer, sampling)
    # else:
    #     sampled = layer
    out = np.zeros(sampled.shape[0]*sampled.shape[1])
    indx = 0
    for i in range(0, sampled.shape[0], 8):
        for j in range(0, sampled.shape[1], 8):
            slice = sampled[i:i+8, j:j+8]
            slice = slice.astype(np.int) - 128
            slice = dct2(slice)
            slice = np.round(slice/q).astype(int)
            out[indx:indx+64] = zigzag(slice)
            indx += 64
    #rle
    return encode_rle(out)

def decompress(layer, sampling, q, info, is_y):

    layer = decore_rle(layer)

    if sampling == "4:2:2":
        out = np.zeros((info.x, int(info.y/2)))
    else:
        out = np.zeros((info.x, int(info.y)))

    for idx, i in enumerate(range(0, layer.shape[0], 64)):
        slice = zigzag(layer[i:i+64])
        slice = slice * q.astype(int)
        slice = idct2(slice)
        slice = slice.astype(int) + 128
        x = (idx*8) % out.shape[1]
        y = int((idx*8)/out.shape[1])*8
        out[y:y+8, x:x+8] = slice

    unsampled = chroma_resample(out, sampling)

    # if not is_y:
    #     unsampled = chroma_resample(out, sampling)
    # else:
    #     unsampled = out

    return unsampled

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

    return encoded_data[:pair_index].astype(int)
    

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

def encode_jpeg(_image, sampling, qy, qc):

    image = _image.copy()

    YCrCb=cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(int)

    Y, Cr, Cb = cv2.split(YCrCb)

    y_compressed = compress(Y, sampling, qy, True)
    cr_compressed = compress(Cr, sampling, qc, False)
    cb_compressed = compress(Cb, sampling, qc, False)

    info = ImageInfo(YCrCb.shape[0], YCrCb.shape[1])

    return y_compressed, cr_compressed, cb_compressed, info

def decode_jpeg(y_compressed, cr_compressed, cb_compressed, sampling, qy, qc, info):
    
    y_decompressed = decompress(y_compressed, sampling, qy, info, True)
    cr_decompressed = decompress(cr_compressed, sampling, qc, info, False)
    cb_decompressed = decompress(cb_compressed, sampling, qc, info, False)

    y_decompressed = np.clip(y_decompressed, 0, 255)

    decompressed = np.dstack([y_decompressed,cr_decompressed,cb_decompressed])
    decompressed=cv2.cvtColor(decompressed.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

    return decompressed, y_decompressed, cr_decompressed, cb_decompressed

if __name__ == "__main__":

    x = 500
    y = 500

    s = 64

    ones = np.ones((8, 8))

    img = cv2.imread("Lab08/firewatch.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.copy()[x:x+s, y:y+s]

    qy_type = QY
    qc_type = QC
    sampling = "4:2:2"

    y_compressed, cr_compressed, cb_compressed, info = encode_jpeg(img, sampling, qy_type, qc_type)
    decompressed, y_decompressed, cr_decompressed, cb_decompressed = decode_jpeg(y_compressed, cr_compressed, cb_compressed, sampling, qy_type, qc_type, info)

    YCrCb=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb).astype(int)

    Y, Cr, Cb = cv2.split(YCrCb)


    # Y = Y[x:x+s, y:y+s]
    # Cr = Cr[x:x+s, y:y+s]
    # Cb = Cb[x:x+s, y:y+s]

    # y_compressed = compress(Y, "4:2:2", QY)
    # cr_compressed = compress(Cr, "4:2:2", QC)
    # cb_compressed = compress(Cb, "4:2:2", QY)

    # info = ImageInfo(s, s)

    # y_decompressed = decompress(y_compressed, "4:2:2", QY, info)
    # cr_decompressed = decompress(cr_compressed, "4:2:2", QC, info)
    # cb_decompressed = decompress(cb_compressed, "4:2:2", QC, info)

    # y_decompressed = np.clip(y_decompressed, 0, 255)

    # decompressed = np.dstack([y_decompressed,cr_decompressed,cb_decompressed])

    # decompressed=cv2.cvtColor(decompressed.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

    # enc, info = encode_jpeg(img, "4:2:2")
    # dec = decode_jpeg(enc, "4:2:2", info)
    
    # decompressed, y_decompressed, cr_decompressed, cb_decompressed = cv2.split(dec)

    fig, axs = plt.subplots(1, 4, sharey=True)
    fig.set_size_inches(9,13)
    axs[0].imshow(img)
    axs[1].imshow(Y,cmap=plt.cm.gray)
    axs[2].imshow(Cr,cmap=plt.cm.gray)
    axs[3].imshow(Cb,cmap=plt.cm.gray)
    plt.show()

    fig, axs = plt.subplots(1, 4, sharey=True)
    fig.set_size_inches(9,13)
    axs[0].imshow(decompressed)
    axs[1].imshow(y_decompressed,cmap=plt.cm.gray)
    axs[2].imshow(cr_decompressed,cmap=plt.cm.gray)
    axs[3].imshow(cb_decompressed,cmap=plt.cm.gray)
    plt.show()
