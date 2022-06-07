
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse

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
    return structural_similarity(K.astype(int), I.astype(int))



    

'''WCZYTANIE OBRAZU'''

_img=cv2.imread('Lab10/flowers.jpg')
img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

# compressed = compress_jpg(img, 70)

# blurred = blur_mean(img, 5)
# blurred = blur_gaussian(img, 5, 0)
# blurred = blur_median(img, 5)
# blurred = blur_bilateral(img, 5, 90, 90)

# noise = noise_SnP(img)
# noise = noise_gauss(img, 0.9)
noise = noise_vals(img, 0.01)
# noise = noise_rand(img, 0.9)

print("mse", mse(noise, img))
# print(mean_squared_error(img, noise))
print("nmse", nmse(noise, img))
print("psnr", psnr(noise, img))
print("if", if0(noise, img))
# print(ssim(noise, img))

# cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
cv2.imwrite('Lab10/0000_3.png', cv2.cvtColor(noise, cv2.COLOR_RGB2BGR))

'''WIZUALIZACJA'''

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].imshow(img)
axs[1].imshow(noise)
plt.show()