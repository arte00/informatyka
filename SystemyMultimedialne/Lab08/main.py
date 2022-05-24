
from re import L
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.fftpack

class JPEG:

    def __init__(self) -> None:
        self.shape = None

def load_image(path, infilename) :
    img = Image.open(path + infilename)
    img.load()
    data = np.asarray(img, dtype=np.uint8)
    return data

def chroma_subsampling(_data, sampling):
    data = _data

    new_data = np.empty((data.shape[0], int(data.shape[1]/2)))
    
    if sampling == "4:4:4":
        new_data = data
    elif sampling == "4:2:2":
        # new_data = data[:, 0:data.shape[1]:2]
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1], 2):
                new_data[row][int(col/2)] = data[row][col]

    return new_data

def chroma_resampling(_data, sampling):
    data = _data
    new_data = np.zeros((data.shape[0], data.shape[1]*2, data.shape[2]), dtype=np.int16)

    if sampling == "4:4:4":
        new_data = data
    elif sampling == "4:2:2":
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_data[i, j*2] = data[i, j]
                new_data[i, j*2+1] = data[i, j]
            
    return new_data

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


def compress(_data, sampling, qTable):

    data = _data.copy()

    print(data)
    data = chroma_subsampling(data, sampling)
    data = data.astype(int) - 128
    print(data)
    data = dct2(data)

    out = np.zeros(_data.shape[0]*_data.shape[1])

    indx = 0
    for row in range(0, data.shape[0], 8):
        for col in range(0, data.shape[1], 8):

            zz = data[row:row+8, col:col+8]
            temp = zigzag(zz)
            out[indx:indx+64] = np.round(temp/qTable.flatten()).astype(int)
            indx += 64

    return out

def decompress(_data, sampling, qTable):
    data = _data

    if sampling ==  "4:2:2":
        out = np.zeros((128, 128))
    else:
        out = np.zeros((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0]))))

    for idx, i in enumerate(range(0, data.shape[0], 64)):

        dequantized = data[i:i+64] * qTable.flatten()
        unzigzaged = zigzag(dequantized)

        x = (idx*8) % out.shape[1]
        y = int((idx*8)/out.shape[1])*8
        out[y:y+8, x:x+8] = unzigzaged

    data = idct2(data)
    data = data.astype(int) + 128
    data = chroma_resampling(data, sampling)
    # data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    return data

def test(data, sampling, qtable):
    compressed = compress(data, sampling, qtable)
    return decompress(compressed, sampling, qtable)


if __name__ == "__main__":

    path = 'Lab08/'
    x = 1500
    y = 1500
    data = load_image(path, 'test3.jpg')[y:y+128, x:x+128]
    # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    
    data=cv2.cvtColor(data,cv2.COLOR_RGB2YCrCb).astype(int)
    data = np.clip(data, 0, 255)
    Y = data[:, :, 0]
    Cr = data[:, :, 1]
    Cb = data[:, :, 2]

    compressed = compress(Y, "4:4:2", QY)

    YCrCb=np.dstack([Y,Cr,Cb]).astype(np.uint8)

    data=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)


    # fig, axs = plt.subplots(4, 1 , sharey=True)
    # fig.set_size_inches(9,13)
    # axs[0].imshow(data)
    # axs[1].imshow(Y,cmap=plt.cm.gray)
    # axs[2].imshow(Cr,cmap=plt.cm.gray)
    # axs[3].imshow(Cb,cmap=plt.cm.gray)
    # plt.show()

    # axs[0,0].imshow(data)
    # axs[1,0].imshow(Y, cmap=plt.cm.gray)
    # axs[2,0].imshow(Cr, cmap=plt.cm.gray)
    # axs[3,0].imshow(Cb, cmap=plt.cm.gray)

    # sampling = "4:2:2"
    # one = False
    # Y2 = test(Y, sampling, QY)
    # Cr2 = test(Cr, QC)
    # Cb2 = test(Cb, QC)

    # axs[0,1].imshow(np.dstack([Y2,Cr2,Cb2]).astype(np.uint8))
    # axs[1,1].imshow(Y2, cmap=plt.cm.gray)
    # axs[2,1].imshow(Cr2, cmap=plt.cm.gray)
    # axs[3,1].imshow(Cb2, cmap=plt.cm.gray)

    plt.show()


