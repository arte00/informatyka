import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

data, fs = sf.read('sound1.wav', dtype='float32')
print(data.dtype)
print(data.shape)
print(data[15000:15050, 0])
print(fs)


new_data = data.copy()

# left channel
sf.write('sound_L.wav', new_data[:, 0], fs)

# right channel
sf.write('sound_R.wav', new_data[:, 1], fs)

# mix two channels
# (right_channel + left_channel) / 2
new_data[:, 0] += new_data[:, 1]
new_data[:, 0] /= 2
new_data[:, 1] = new_data[:, 0]
sf.write('sound_mix.wav', new_data, fs)

sd.play(new_data, fs)
status = sd.wait()

plt.subplot(2, 1, 1)
plt.plot(data[:, 0])
plt.show()
