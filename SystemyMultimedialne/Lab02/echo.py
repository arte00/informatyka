try:
    import pyaudio
    import numpy as np
    import pylab
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    import time
    import sys
    import wave
except:
    print("Something didn't import")

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 44100
CHUNK = 1024 # 1024bytes of data red from a buffer
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "file.wav"

def process_data(in_data, frame_count, time_info, flag):
    delay = 1
    global Frame_buffer, frame_idx
    in_audio_data = np.frombuffer(in_data, dtype=np.int16)
    Frame_buffer[frame_idx:(frame_idx+CHUNK),0]=in_audio_data
    ################################
    ## Do something wih data
    out_audio_data = in_audio_data
    ################################
    Frame_buffer[frame_idx:(frame_idx+CHUNK),1]=out_audio_data
    out_data =  out_audio_data.tobytes()
    frame_idx+=CHUNK
    return Frame_buffer, pyaudio.paContinue

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(input_device_index =1,
                    output_device_index=3,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=process_data)

global Frame_buffer,frame_idx

N=10
Frame_buffer = np.zeros(((N+1)*RATE,2))
frame_idx=0

stream.start_stream()
while stream.is_active():
    time.sleep(N)
    stream.stop_stream()
stream.close()

plt.subplot(2,1,1)
plt.plot(np.arange(len(Frame_buffer[:, 0])),Frame_buffer[:,0])
plt.subplot(2,1,2)
plt.plot(np.arange(len(Frame_buffer[:, 1])),Frame_buffer[:,1])
plt.show()

