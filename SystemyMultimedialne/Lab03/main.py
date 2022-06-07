from re import S
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def check_if_grayscale(data):
    if (len(data.shape)<3): 
        return True
    else:
        return False


def load_image(path, infilename) :
    img = Image.open(path + infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename) :
    img = Image.fromarray(np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save(outfilename)


def nearest_neightbour_interpolation(_data, rescale):
    data = _data

    width = data.shape[1]
    height = data.shape[0]
    
    if len(data.shape) > 2:
        colors = data.shape[2]

    is_grayscale = check_if_grayscale(data)
    new_width = np.int32(width * rescale)
    new_height = np.int32(height * rescale)

    if is_grayscale:
        output = np.zeros(new_width*new_height, dtype="int32").reshape(new_height, new_width)
    else:
        output = np.zeros(new_width*new_height*colors, dtype="int32").reshape(new_height, new_width, colors)

    for x in range(0, new_width):
        for y in range(0, new_height):
            data_x = np.int32(round(float(x) / float(new_width) * float(width)))
            data_y = np.int32(round(float(y) / float(new_height) * float(height)))

            data_x = min(data_x, width-1)
            data_y = min(data_y, height-1)

            if is_grayscale:
                output[x][y] = data[data_x][data_y]
            else:
                output[y][x][0] = data[data_y][data_x][0]
                output[y][x][1] = data[data_y][data_x][1]
                output[y][x][2] = data[data_y][data_x][2]

    return output


def present_nni():
    image = load_image('parthenon.jpg')

    print(image.shape)

    nni = nearest_neightbour_interpolation(image, 0.1)
    print("/n")

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(122)
    plt.imshow(nni, cmap='gray', vmin=0, vmax=255)
    plt.show()


def bilinear_interpolation(_data, rescale):
    data = _data

    width = data.shape[1]
    height = data.shape[0]

    if len(data.shape) > 2:
        colors = data.shape[2]

    is_grayscale = check_if_grayscale(data)
    new_width = np.int32(width * rescale)
    new_height = np.int32(height * rescale)

    if is_grayscale:
        output = np.zeros(new_width*new_height, dtype="int32").reshape(new_height, new_width)
    else:
        output = np.zeros(new_width*new_height*colors, dtype="int32").reshape(new_height, new_width, colors)

    for i in range(0, new_height):
        for j in range(0, new_width):

            x = j / new_width * width
            y = i / new_height * height

            x_prev = int(np.floor(x))
            x_next = x_prev + 1
            y_prev = int(np.floor(y))
            y_next = y_prev + 1

            x_prev = min(x_prev, width - 1)
            x_next = min(x_next, width - 1)
            y_prev = min(y_prev, height - 1)
            y_next = min(y_next, height - 1)

            if is_grayscale:
                output[i][j][c] = (1. - (y_next - y)) * (data[y_next][x_prev][c] * (x_next - x) + data[y_next][x_next][c] * (1. - (x_next - x))) \
                + (y_next - y) * (data[y_prev][x_prev][c] * (x_next - x) + data[y_prev][x_next][c] * (1. - (x_next - x)))
            else:
                for c in range(3):
                    output[i][j][c] = (1. - (y_next - y)) * (data[y_next][x_prev][c] * (x_next - x) + data[y_next][x_next][c] * (1. - (x_next - x))) \
                    + (y_next - y) * (data[y_prev][x_prev][c] * (x_next - x) + data[y_prev][x_next][c] * (1. - (x_next - x)))

    return output

def scale_mean(_data, rescale):
    data = _data

    width = data.shape[1]
    height = data.shape[0]

    if len(data.shape) > 2:
        colors = data.shape[2]

    is_grayscale = check_if_grayscale(data)
    new_width = np.int32(width * rescale)
    new_height = np.int32(height * rescale)

    if is_grayscale:
        output = np.zeros(new_width*new_height, dtype="int32").reshape(new_height, new_width)
    else:
        output = np.zeros(new_width*new_height*colors, dtype="int32").reshape(new_height, new_width, colors)

    chunk_width = int(np.ceil(1/rescale))
    chunk_height = int(np.ceil(1/rescale))

    for y in range(new_height):
        for x in range(new_width):

            box_start_x = int(np.floor(x/rescale))
            box_start_y = int(np.floor(y/rescale))

            x_end = min(box_start_x + chunk_width, width-1)
            y_end = min(box_start_y + chunk_height, height-1)

            output[y][x] = np.mean(data[box_start_y:y_end,box_start_x:x_end], axis=(0, 1))

    return output

def scale_weighted_mean(_data, rescale):
    data = _data

    width = data.shape[1]
    height = data.shape[0]

    if len(data.shape) > 2:
        colors = data.shape[2]

    is_grayscale = check_if_grayscale(data)
    new_width = np.int32(width * rescale)
    new_height = np.int32(height * rescale)

    if is_grayscale:
        output = np.zeros(new_width*new_height, dtype="int32").reshape(new_height, new_width)
    else:
        output = np.zeros(new_width*new_height*colors, dtype="int32").reshape(new_height, new_width, colors)

    box_width = int(np.ceil(1/rescale))
    box_height = int(np.ceil(1/rescale))

    for y in range(new_height):
        for x in range(new_width):

            x_start = int(np.floor(x/rescale))
            y_start = int(np.floor(y/rescale))

            x_end = min(x_start + box_width, width-1)
            y_end = min(y_start + box_height, height-1)

            pixels = np.array(data[y_start:y_end,x_start:x_end])

            x_weights = np.linspace(0.5, 1, box_width)
            y_weights = np.linspace(1, 0.5, box_height)

            weights = np.array([[[x_weights[i]*y_weights[j] for _ in range(3)] for i in range((x_end - x_start))] for j in range((y_end - y_start))])

            box = pixels * weights

            pixel = np.uint8(np.sum(box, axis=(0, 1)) / np.sum(weights, axis=(0, 1)))

            output[y][x] = pixel

            if y % 100 == 0 and x % 100 == 0:
                print(str(y) + "/" + str(new_height))
                print(str(x) + "/" + str(new_width))

    return output

def scale_median(_data, rescale):
    data = _data

    width = data.shape[1]
    height = data.shape[0]

    if len(data.shape) > 2:
        colors = data.shape[2]

    is_grayscale = check_if_grayscale(data)
    new_width = np.int32(width * rescale)
    new_height = np.int32(height * rescale)

    if is_grayscale:
        output = np.zeros(new_width*new_height, dtype="int32").reshape(new_height, new_width)
    else:
        output = np.zeros(new_width*new_height*colors, dtype="int32").reshape(new_height, new_width, colors)

    box_width = int(np.ceil(1/rescale))
    box_height = int(np.ceil(1/rescale))

    for y in range(new_height):
        for x in range(new_width):

            box_start_x = np.int32(np.floor(x/rescale))
            box_start_y = np.int32(np.floor(y/rescale))

            x_end = min(box_start_x + box_width, width-1)
            y_end = min(box_start_y + box_height, height-1)

            output[y][x] = np.median(data[box_start_y:y_end,box_start_x:x_end], axis=(0, 1))

    return output


if __name__ == "__main__":

    image = load_image('Lab03/', 'SM_Lab03/0008.tif')
    scale = 1.33

    # weigted_mean = scale_weighted_mean(image, scale)
    # edges_median = cv2.Canny(np.uint8(weigted_mean), 100, 100)

    # plt.figure(1)
    # plt.subplot(131)
    # plt.title("original")
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(132)
    # plt.title("weighted mean")
    # plt.imshow(weigted_mean, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(133)
    # plt.title("mean")
    # plt.imshow(edges_median, cmap='gray', vmin=0, vmax=255)
    # plt.show()


    # result = scale_mean(image, scale)
    # print(result.shape)
    # plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # powiększanie

    # nni = nearest_neightbour_interpolation(image, scale)
    # bilinear = bilinear_interpolation(image, scale)

    # plt.figure(1)
    # plt.subplot(131)
    # plt.title("original")
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(132)
    # plt.title("nearest")
    # plt.imshow(nni, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(133)
    # plt.title("bilinear")
    # plt.imshow(bilinear, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # pomniejszanie

    # mean = scale_mean(image, scale)
    # median = scale_median(image, scale)

    # plt.figure(1)
    # plt.subplot(131)
    # plt.title("original")
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(132)
    # plt.title("mean")
    # plt.imshow(mean, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(133)
    # plt.title("median")
    # plt.imshow(median, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # nni = nearest_neightbour_interpolation(image, scale)
    # bilinear = bilinear_interpolation(image, scale)
    # mean = scale_mean(image, scale)
    # median = scale_median(image, scale)

    # edges_og = cv2.Canny(np.uint8(image), 100, 100)
    # edges_nni = cv2.Canny(np.uint8(nni), 100, 100)
    # edges_bilinear = cv2.Canny(np.uint8(bilinear), 100, 100)
    # edges_mean = cv2.Canny(np.uint8(mean), 100, 100)
    # edges_median = cv2.Canny(np.uint8(median), 100, 100)

    # plt.title("original")
    # plt.imshow(edges_og, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.title("mean")
    # plt.imshow(edges_nni, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.title("median")
    # plt.imshow(edges_bilinear, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.title("nearest")
    # plt.imshow(edges_mean, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.title("bilinear")
    # plt.imshow(edges_median, cmap='gray', 1, 4, 5, 6, 7)
    # plt.show()







