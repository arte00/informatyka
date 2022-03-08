import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

data, fs = sf.read('sound1.wav', dtype='float32')
print(data[100])
data[:, [0, 1]] += np.random.normal(0, 0.01, data.shape)
print(data[100])

sd.play(data, fs)


plt.subplot(2, 1, 1)
plt.plot(data[:, 0])
plt.show()

status = sd.wait()
