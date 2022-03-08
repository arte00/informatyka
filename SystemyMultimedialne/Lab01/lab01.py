import numpy as np
import matplotlib.pyplot as plt


class Lab01:

    def __init__(self, img, w1=-1, w2=-1, k1=-1, k2=-1):
        self.w1 = 0 if w1 == -1 else w1
        self.w2 = img.shape[0] if w1 == -1 else w2
        self.k1 = 0 if w1 == -1 else w1
        self.k2 = img.shape[1] if w1 == -1 else k2
        self.img = img[self.w1:self.w2, self.k1:self.k2, :]

    def show_plot(self):

        y1 = 0.299 * self.img[:, :, 0] + 0.587 * self.img[:, :, 1] + 0.114 * self.img[:, :, 2]
        y2 = 0.2126 * self.img[:, :, 0] + 0.7152 * self.img[:, :, 1] + 0.0722 * self.img[:, :, 2]

        # I row

        plt.subplot(3, 3, 1)
        print(self.img.shape)
        plt.imshow(self.img)

        plt.subplot(3, 3, 2)
        plt.imshow(y1, cmap=plt.cm.gray)

        plt.subplot(3, 3, 3)
        plt.imshow(y2, cmap=plt.cm.gray)

        # II row

        plt.subplot(3, 3, 4)
        plt.imshow(self.img[:, :, 0], cmap=plt.cm.gray)

        plt.subplot(3, 3, 5)
        plt.imshow(self.img[:, :, 1], cmap=plt.cm.gray)

        plt.subplot(3, 3, 6)
        plt.imshow(self.img[:, :, 2], cmap=plt.cm.gray)

        # III row

        plt.subplot(3, 3, 7)
        img_red = self.img.copy()
        img_red[:, :, 1:] = 0
        plt.imshow(img_red)

        plt.subplot(3, 3, 8)
        img_green = self.img.copy()
        img_green[:, :, 0] = 0
        img_green[:, :, 2] = 0
        plt.imshow(img_green)

        plt.subplot(3, 3, 9)
        img_blue = self.img.copy()
        img_blue[:, :, :2] = 0
        plt.imshow(img_blue)

        plt.show()


"""EXC 1"""

img1 = plt.imread('pic1.png')
pic1 = Lab01(img1)
pic1.show_plot()

"""EXC 2"""

img2 = plt.imread('pic2.jpg')
pic2 = Lab01(img2, 0, 200, 0, 200)
pic2.show_plot()


