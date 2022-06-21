import numpy as np
import matplotlib.pyplot as plt
import cv2

data = np.array([[np.random.randint(0, 2) for _ in range(33)] for _ in range(33)])

# print(data)


def make_position_pattern(_data):

    data = _data.copy()
    chunk1 = [[1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    chunk2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 1, 1, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0]]

    chunk3 = [[0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 0, 0, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 0, 0, 0, 0, 0, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    data[0:9, 0:9] = chunk1
    data[-9:, 0:9] = chunk2
    data[0:9, -9:] = chunk3

    return data

def write_data(_data):

    data = _data.copy()

    return data




data = make_position_pattern(data)
plt.imshow(data, cmap='Greys')
plt.show()