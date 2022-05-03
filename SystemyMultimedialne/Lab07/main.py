import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def quantize(_data, _bits):

    assert 2 <= _bits <= 32

    out_val = _data.astype(np.float32)
    _start = 0
    _end = 0

    if np.issubdtype(_data.dtype, np.floating):
        _start = -1
        _end = 1
    else:
        _start = np.iinfo(_data.dtype).min
        _end   = np.iinfo(_data.dtype).max

    _range2 = 2**_bits-1

    out_val = (out_val-_start)/(_end-_start)
    out_val = np.round(out_val*_range2)/_range2
    out_val = ((out_val*(_end-_start))+_start)

    return out_val.astype(_data.dtype)

def plot_sound_diff(_data, _data1, _data2, _freq, _time_ms, _title="???"):
    _max = int(_freq*(_time_ms/1000))
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, _max)/_freq, _data[0:_max], label="Original")
    ax.plot(np.arange(0, _max) / _freq, _data1[0:_max], label="a")
    ax.plot(np.arange(0, _max) / _freq, _data2[0:_max], label="mu")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.title(_title)
    plt.show()


def encode_a_law(data):

    a = []
    for i in data:
        if np.abs(i) < 1 / 87.6:
            x = np.sign(i) * ((87.6 * np.abs(i)) / (1 + np.log(87.6)))
        elif 1 / 87.6 <= np.abs(i) <= 1:
            x = np.sign(i) * ((1 + np.log(87.6 * np.abs(i))) / (1 + np.log(87.6)))
        a.append(x)

    return np.array(a)

def decode_a_law(data):

    a = []
    for i in data:
        if np.abs(i) < (1 / (1 + np.log(87.6))):
            x = np.sign(i) * ((np.abs(i) * (1 + np.log(87.6))) / 87.6)
        elif 1 / (1 + np.log(87.6)) <= np.abs(i) <= 1:
            x = np.sign(i) * (np.exp(np.abs(i)*(1 + (np.log(87.6))) - 1)/87.6)
        a.append(x)

    return np.array(a)

def a_law(data, bits):

    out = encode_a_law(data)
    out = quantize(out, bits)
    out = decode_a_law(out)

    return out

data, freq = sf.read('Lab07/sing_medium1.wav', dtype=np.float32)

_time_ms = 2000
_max = int(freq*(_time_ms/1000))

if len(data.shape) > 1:
    data = data[:, 0]

data1 = a_law(data, 10)

plot_sound_diff(data, data1, data1, freq, 7, "ess")

sd.play(data[0:_max], samplerate=freq, blocking=True)
sd.play(data1[0:_max], samplerate=freq, blocking=True)