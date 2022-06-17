
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse
import pandas as pd
from sklearn.linear_model import LinearRegression

'''KOMPRESJA DO JPEG'''

def compress_jpg(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

'''ROZMYWANIE'''

def blur_mean(img, box_size):
    return cv2.blur(img, (box_size, box_size))

def blur_gaussian(img, box_size, sigma):
    return cv2.GaussianBlur(img, (box_size, box_size), sigma)

def blur_median(img, box_size):
    return cv2.medianBlur(img, box_size)

def blur_bilateral(img, box_size, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, box_size, sigma_color, sigma_space)

'''ZASZUMIENIE'''

def noise_gauss(img, alpha):
    gauss = np.random.normal(0,25,(img.shape))
    noisy = (img + alpha * gauss).clip(0,255).astype(np.uint8)
    return noisy

def noise_vals(img, alpha):
    vals = len(np.unique(img))
    vals =  alpha * 2 ** np.ceil(np.log2(vals))
    noisy = (np.random.poisson(img * vals) / float(vals)).clip(0,255).astype(np.uint8)
    return noisy

def noise_rand(img, alpha):
    rand = 25*np.random.random((img.shape))
    noisy = (img + alpha * rand).clip(0,255).astype(np.uint8)
    return noisy

'''SÓL I PIEPRZ'''

def noise_SnP(_img, S=255, P=0, rnd=(333,9999)):
    img = _img.copy()
    r, c = img.shape[0:2]
    number_of_pixels = random.randint(rnd[0], rnd[1])
    for i in range(number_of_pixels):
        y=random.randint(0, r - 1)
        x=random.randint(0, c - 1)
        img[y][x] = S
    number_of_pixels = random.randint(rnd[0], rnd[1])
    for i in range(number_of_pixels):
        y=random.randint(0, r - 1)
        x=random.randint(0, c - 1)
        img[y][x] = P
    return img

'''OBIEKTYWNE MIARY JAKOŚCI'''

def mse(K, I):

    # s1 = 1 / (K.shape[0] * K.shape[1] * K.shape[2]) * np.sum(np.power((I.astype(np.int32) - K.astype(np.int32)), 2))

    s1 = 0

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                s1 += (int(I[i][j][k]) - int(K[i][j][k]))**2

    return 1 / (K.shape[0] * K.shape[1] * K.shape[2]) * s1

    # return s1


def nmse(K, I):
    
    # s = 1 / (K.shape[0] * K.shape[1] * K.shape[2]) * np.sum(np.power((I.astype(np.int32) - K.astype(np.int32)), 2))

    s1 = 0

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                s1 += (int(I[i][j][k]) - int(K[i][j][k]))**2

    sk = 0

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                sk += (int(K[i][j][k]))**2

    return s1 / sk

def psnr(K, I):

    MAXI = 255

    # s0 = 1 / (K.shape[0] * K.shape[1] * K.shape[2]) * np.sum(np.power((I.astype(np.int32) - K.astype(np.int32)), 2))
    s0 = mse(K, I)
    s1 = 10 * np.log10((MAXI**2 / s0))

    return s1

def if0(K, I):

    # s0 = np.sum(np.power((K.astype(np.int32) - I.astype(np.int32)), 2))
    # s1 = np.sum(np.power((K.astype(np.int32)*I.astype(np.int32)), 2))

    s0 = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                s0 += (int(K[i][j][k]) - int(I[i][j][k]))**2

    s1 = 0

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                s1 += (int(K[i][j][k])*int(I[i][j][k]))

    return 1 - (s0/s1)

# to do
def ssim(K, I):
    return structural_similarity(K, I, channel_axis=2)



    

'''WCZYTANIE OBRAZU'''

_img=cv2.imread('Lab10/cake.jpg')
img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

# compressed = compress_jpg(img, 70)

# edited = blur_mean(img, 5)
# blurred = blur_gaussian(img, 5, 0)
# blurred = blur_median(img, 5)
# blurred = blur_bilateral(img, 5, 90, 90)

# noise = noise_SnP(img)
# noise = noise_gauss(img, 0.9)
# noise = noise_vals(img, 0.01)
# noise = noise_rand(img, 0.9)

# edited = compress_jpg(img, 90)
# edited = compress_jpg(img, 80)
# edited = compress_jpg(img, 70)
# edited = compress_jpg(img, 60)
# edited = compress_jpg(img, 50)
# edited = compress_jpg(img, 40)
# edited = compress_jpg(img, 30)
# edited = compress_jpg(img, 25)
# edited = compress_jpg(img, 15)
# edited = compress_jpg(img, 5)


# edited = blur_mean(img, 5)
# edited = blur_mean(img, 15)
# edited = blur_gaussian(img, 5, 5)
# edited = blur_gaussian(img, 5, 0)
# edited = blur_gaussian(img, 19, 11)
# edited = blur_median(img, 5)
# edited = blur_median(img, 21)
# edited = blur_median(img, 15)
# edited = blur_bilateral(img, 5, 90, 90)
# edited = blur_bilateral(img, 3, 80, 80)

# edited = noise_SnP(img)
# edited = noise_gauss(img, 0.3)
# edited = noise_gauss(img, 0.65)
# edited = noise_gauss(img, 0.7)
# edited = noise_gauss(img, 0.9)
# edited = noise_vals(img, 0.1)
# edited = noise_vals(img, 0.005)
# edited = noise_rand(img, 0.9)
# edited = noise_rand(img, 0.65)
# edited = noise_rand(img, 0.45)

# print(mse(edited1, img))
# print(mse(edited2, img))
# print(mse(edited3, img))
# print(mse(edited4, img))
# print(mse(edited5, img))
# print(mse(edited6, img))
# print(mse(edited7, img))
# print(mse(edited8, img))
# print(mse(edited9, img))
# print(mse(edited10, img))

# edited = edited10

# print("mse: ", mse(edited, img))
# print("nmse: ", nmse(edited, img))
# print("psnr: ", psnr(edited, img))
# print("if: ", if0(edited, img))
# print("ssim: ", ssim(edited, img))

# cv2.cvtColor(edited, cv2.COLOR_BGR2RGB)
# cv2.imwrite('Lab10/0002_10.jpg', cv2.cvtColor(edited, cv2.COLOR_RGB2BGR))

# '''WIZUALIZACJA'''

# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].imshow(img)
# axs[1].imshow(edited)
# plt.show()

results = pd.read_csv("Lab10/kwiat - jpg.csv")

imgs = []
for i in range(10):
    imgs.append("img" + str(i))


# edited = compress_jpg(img, 90)
# edited = compress_jpg(img, 80)
# edited = compress_jpg(img, 70)
# edited = compress_jpg(img, 60)
# edited = compress_jpg(img, 50)
# edited = compress_jpg(img, 40)
# edited = compress_jpg(img, 30)
# edited = compress_jpg(img, 25)
# edited = compress_jpg(img, 15)
# edited = compress_jpg(img, 5)

norms = [
    ssim(compress_jpg(img, 90), img),
    ssim(compress_jpg(img, 80), img),
    ssim(compress_jpg(img, 70), img),
    ssim(compress_jpg(img, 60), img),
    ssim(compress_jpg(img, 50), img),
    ssim(compress_jpg(img, 40), img),
    ssim(compress_jpg(img, 30), img),
    ssim(compress_jpg(img, 25), img),
    ssim(compress_jpg(img, 15), img),
    ssim(compress_jpg(img, 5), img),
]

# norms = [
#     mse(edited1, img),
#     mse(edited2, img),
#     mse(edited3, img),
#     mse(edited4, img),
#     mse(edited5, img),
#     mse(edited6, img),
#     mse(edited7, img),
#     mse(edited8, img),
#     mse(edited9, img),
#     mse(edited10, img),
# ]


# norms = [2209.044383, 56.369219, 256.741198, 256.741198, 475.272304, 7.552477, 143.069546, 155.411983, 79.662363, 36.820629]



if __name__ == "__main__":

    # results = pd.read_csv("Lab10/ciasto - szum.csv")

    # imgs = []
    # for i in range(10):
    #     imgs.append("img" + str(i))


    base=pd.DataFrame(data=results).transpose()
    badani = base.iloc[1]
    base = base.drop(base.index[[0, 1]])


    base.index = imgs

    base.assign(Name="norms")
    base.insert(0, "norms", norms)

    col = ["norms"]
    for badany in badani:
        col.append(badany)
    base.columns = col

    base = base.reindex(sorted(base.columns), axis=1)

    mos = []

    for i in range(len(imgs)):
        mos.append(list(base.iloc[i, :-1].values))

    print(mos)

    # symbol = []

    print(base)

    # for _ in range(3):
    #     symbol.append("r*")
    # for _ in range(3):
    #     symbol.append("g^")
    # for _ in range(3):
    #     symbol.append("bv")
 
    for i in range(len(imgs)):
        plt.plot(i, mos[i][0], "r*")
        plt.plot(i, mos[i][1], "r*")
        plt.plot(i, mos[i][2], "r*")

        plt.plot(i, mos[i][3], "g^")
        plt.plot(i, mos[i][4], "g^")
        plt.plot(i, mos[i][5], "g^")

        plt.plot(i, mos[i][6], "bv")
        plt.plot(i, mos[i][7], "bv")
        plt.plot(i, mos[i][8], "bv")

        plt.plot(i, mos[i][9], "gv")
        plt.plot(i, mos[i][10], "gv")
        plt.plot(i, mos[i][11], "gv")

        plt.plot(i, mos[i][12], "rv")
        plt.plot(i, mos[i][13], "rv")
        plt.plot(i, mos[i][14], "rv")

        plt.plot(i, mos[i][15], "b*")
        plt.plot(i, mos[i][16], "b*")
        plt.plot(i, mos[i][17], "b*")

        plt.plot(i, mos[i][18], "g*")
        plt.plot(i, mos[i][19], "g*")
        plt.plot(i, mos[i][20], "g*")
        
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """agregated dla badanego"""

    for i in range(len(imgs)):
        plt.plot(i, np.sum(mos[i][0:3])/3, "r*")
        plt.plot(i, np.sum(mos[i][3:6])/3, "g^")
        plt.plot(i, np.sum(mos[i][6:9])/3, "bv")
        plt.plot(i, np.sum(mos[i][9:12])/3, "gv")
        plt.plot(i, np.sum(mos[i][12:15])/3, "rv")
        plt.plot(i, np.sum(mos[i][15:18])/3, "b*")
        plt.plot(i, np.sum(mos[i][18:-1])/3, "g*")
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """agreggated"""
    for i in range(len(imgs)):
        plt.plot(i, np.sum(mos[i][:])/21, "r*")
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """mos + miary"""

    base = base.sort_values(by=['norms'])
    print(base)

    mos = []

    for i in range(len(imgs)):
        mos.append(list(base.iloc[i, :-1].values))

    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], mos[i][0], "r*")
        plt.plot(base.iloc[i, -1], mos[i][1], "r*")
        plt.plot(base.iloc[i, -1], mos[i][2], "r*")
        plt.plot(base.iloc[i, -1], mos[i][3], "g^")
        plt.plot(base.iloc[i, -1], mos[i][4], "g^")
        plt.plot(base.iloc[i, -1], mos[i][5], "g^")
        plt.plot(base.iloc[i, -1], mos[i][6], "bv")
        plt.plot(base.iloc[i, -1], mos[i][7], "bv")
        plt.plot(base.iloc[i, -1], mos[i][8], "bv")
        plt.plot(base.iloc[i, -1], mos[i][9], "gv")
        plt.plot(base.iloc[i, -1], mos[i][10], "gv")
        plt.plot(base.iloc[i, -1], mos[i][11], "gv")

        plt.plot(base.iloc[i, -1], mos[i][12], "rv")
        plt.plot(base.iloc[i, -1], mos[i][13], "rv")
        plt.plot(base.iloc[i, -1], mos[i][14], "rv")

        plt.plot(base.iloc[i, -1], mos[i][15], "b*")
        plt.plot(base.iloc[i, -1], mos[i][16], "b*")
        plt.plot(base.iloc[i, -1], mos[i][17], "b*")

        plt.plot(base.iloc[i, -1], mos[i][18], "g*")
        plt.plot(base.iloc[i, -1], mos[i][19], "g*")
        plt.plot(base.iloc[i, -1], mos[i][20], "g*")
    plt.xticks(base.iloc[:, -1].values)
    plt.show()

    """agregated dla badanego + mos"""

    symbol1 = ["r*", "g^"]
    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], np.sum(mos[i][0:3])/3, "r*")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][3:6])/3, "g^")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][6:9])/3, "bv")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][9:12])/3, "gv")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][12:15])/3, "rv")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][15:18])/3, "b*")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][18:-1])/3, "g*")
    plt.xticks(base.iloc[:, -1].values)
    plt.show()

    """agreggated"""
    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], np.sum(mos[i][:])/21, "r*")
    plt.xticks(base.iloc[:, -1].values)
    plt.show()