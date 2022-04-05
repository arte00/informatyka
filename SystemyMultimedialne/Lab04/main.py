
from cgi import test
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

def check_if_grayscale(data):
    if (len(data.shape)<3): 
        return True
    else:
        return False

def color_fit(_pixel, _palette):

    pixel = _pixel
    palette = _palette

    if len(pixel) == 3:

        r = pixel[0]
        g = pixel[1]
        b = pixel[2]

        color_diffs = np.zeros((1, _palette.shape[1]))

        print(color_diffs)


def dithering_random():
    pass

def dithering_organized():
    pass

def dithering_floyd_steinberg():
    pass

test_palette = [[255*i, 255*j, 255*k] for i in range(2) for j in range(2) for k in range(2)]
test_colors = [[i] for i in test_palette]


if __name__ == '__main__':

    path = 'Lab04/SM_Lab04/'

    image = load_image(path, '0007.tif')

    print(test_palette)

    plt.imshow(test_colors)
    plt.show()

    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.show()