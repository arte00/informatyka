import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def quantize_a_law(data, bits):

    start = -1
    end = 1

    bit_n = 2**bits-1

    data = (data-start)/(end-start)
    data = np.round(data*bit_n)/bit_n
    data = (data*2)-1

    return data


def encode_a_law(data):

    x_data = []
    for i in data:
        if np.abs(i) < 1 / 87.6:
            x = np.sign(i) * ((87.6 * np.abs(i)) / (1 + np.log(87.6)))
        elif 1 / 87.6 <= np.abs(i) <= 1:
            x = np.sign(i) * ((1 + np.log(87.6 * np.abs(i))) / (1 + np.log(87.6)))
        x_data.append(x)

    return np.array(x_data)


def decode_a_law(data):

    y_data = []
    for i in data:
        if np.abs(i) < (1 / (1 + np.log(87.6))):
            y = np.sign(i) * ((np.abs(i) * (1 + np.log(87.6))) / 87.6)
        elif 1 / (1 + np.log(87.6)) <= np.abs(i) <= 1:
            y = np.sign(i) * (np.exp(np.abs(i)*(1 + (np.log(87.6))) - 1)/87.6)
        y_data.append(y)

    return np.array(y_data)


def a_law(data, bits):

    out = encode_a_law(data)
    out = quantize_a_law(out, bits)
    out = decode_a_law(out)

    return out


def encode_dpcm(_data, bits):

    data = _data.copy()

    e = data[0]

    data_min = np.min(data)
    data_max = np.max(data)

    for x in range(1, data.shape[0]):
        diff = data[x] - e
        diff = quantize_dpcm(diff, bits, data_min, data_max)
        data[x] = diff
        e += diff
        
    return data


def decode_dpcm(_data):

    data = _data.copy()

    for x in range(1, data.shape[0]):
        data[x] = data[x-1] + _data[x]

    return data


def quantize_dpcm(_data, bits, data_min, data_max):

    bit_n = 2 ** bits - 1

    data = _data.copy()

    data = (data - data_min) / (data_max - data_min)
    data = np.round(data * bit_n) / bit_n
    data = ((data * (data_max - data_min)) + data_min)

    return data.astype(data.dtype)


def dpcm(data, bits):
    out = encode_dpcm(data, bits)
    out = decode_dpcm(out)
    return out
    

data, frequency = sf.read('Lab07/sing_low1.wav', dtype=np.float32)

if len(data.shape) > 1:
    data = data[:, 0]

bits = 8
time = 2 # only for plot

data1 = a_law(data, bits)
data2 = dpcm(data, bits)

def exc_2(data, data1, frequency, time):
    end = int(frequency*(time/1000))
    plt.plot(np.arange(0, end), data[0:end])
    plt.plot(np.arange(0, end), data1[0:end])
    plt.show()

# exc_2(data, data1, frequency, time)
# exc_2(data, data2, frequency, time)

sd.play(data, samplerate=frequency, blocking=True)
sd.play(data1, samplerate=frequency, blocking=True)
sd.play(data2, samplerate=frequency, blocking=True)