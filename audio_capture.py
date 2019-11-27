import json
import time
from multiprocessing import Queue

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.signal import resample
from threading import Thread
from multiprocessing import Process
from audio_visualiser import start_app

matplotlib.use('TKAGG')  # THIS MAKES IT FAST!

FORMAT = pyaudio.paInt16  # We use 16bit format per sample
CHANNELS = 1
RATE = 48100
CHUNK_SIZE = int(RATE / 60)  # target chunk size for 30FPS


def plot_audio(queued_audio, current_audio, processed_audio):
    plt.clf()
    # plt.subplot(3, 1, 1)
    # plt.plot(queued_audio)
    # plt.subplot(3, 1, 2)
    # plt.plot(current_audio)
    # plt.subplot(3, 1, 3)
    plt.plot(processed_audio)
    plt.xscale('log')
    plt.show()
    plt.pause(0.00001)


def audio_fft(data, trimBy=0, logScale=False, divBy=None):
    # left, right = np.split(np.abs(np.fft.fft(data)), 2)
    # ys = np.add(left, right[::-1])
    ys = np.abs(np.fft.fft(data))
    if logScale:
        ys = np.multiply(20, np.log10(ys))
    # xs = np.arange(CHUNK_SIZE / 2, dtype=float)
    if trimBy:
        i = int((CHUNK_SIZE / 2) / trimBy)
        ys = ys[:i]
        # xs = xs[:i] * RATE / CHUNK_SIZE
    if divBy:
        ys = ys / float(divBy)
    return ys


def dfft_audio_in(audio_data):
    dfft = 10. * np.log10(abs(np.fft.rfft(audio_data, 1999)))
    dfft = np.multiply(20, np.log10(dfft))
    return dfft


def audio_to_numpy(audio_in) -> np.ndarray:
    audio_data = np.frombuffer(audio_in, np.int16).reshape(1, -1)
    return audio_data  # [:int(CHUNK_SIZE / 2)]


def preprocess_audio(audio_data: np.ndarray) -> np.ndarray:
    # audio_data = (audio_data - audio_data.mean()) / audio_data.std()

    # audio_data[audio_data < .95] = 0
    # audio_data = 1 / (1 + np.exp(audio_data))
    audio_data = audio_fft(audio_data)  # [:int(CHUNK_SIZE / 2)]
    # audio_data = np.roll(audio_data[0][:int(audio_data.size/2)].reshape(1, -1), 10)
    # audio_data = audio_data / audio_data.max()
    # audio_data[audio_data < .5] = 0
    # audio_data = audio_data[0, :500]

    if audio_data.size != 1000:
        audio_data = resample(audio_data, 1000, axis=1)
    return audio_data


def send_data(in_data,  # recorded data if input=True; else None
              frame_count,  # number of frames
              time_info,  # dictionary
              status_flags):  # PaCallbackFlags
    st = time.time()
    current_audio = audio_to_numpy(in_data)
    processed_audio = preprocess_audio(current_audio)
    et = time.time()
    print("{}Hz: {}".format(1 / (et - st + 1e-10), frame_count))
    plot_audio([], current_audio[0], processed_audio[0])

    return None, pyaudio.paContinue


def start_audio_cap(gan_audio_queue: Queue, render_audio_queue: Queue):
    audio = pyaudio.PyAudio()
    recording_device_idx = 1
    n_devices = audio.get_device_count()
    for dev_idx in range(n_devices):
        print("Audio Device: {}".format(json.dumps(audio.get_device_info_by_index(dev_idx), indent=4)))
    print("Recoding on: {}".format(json.dumps(audio.get_device_info_by_index(2), indent=4)))
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=recording_device_idx,
                        stream_callback=None)

    stream.start_stream()
    plt.ion()
    queued_audio = [0]
    current_audio = [0]
    vis_thread = Thread(target=start_app, args=[render_audio_queue])
    vis_thread.start()
    while True:
        try:
            current_audio = audio_to_numpy(stream.read(CHUNK_SIZE, exception_on_overflow=False))
            processed_audio = preprocess_audio(current_audio)
            if gan_audio_queue is not None and not gan_audio_queue.full():
                gan_audio_queue.put(processed_audio / processed_audio.max(), False)
                queued_audio = processed_audio
            if render_audio_queue is not None and not render_audio_queue.full():
                render_audio_queue.put(processed_audio.reshape(processed_audio.size), False)
            # plot_audio(queued_audio[0], current_audio[0], processed_audio[0])
        except KeyboardInterrupt:
            stream.stop_stream()
            # vis_thread.terminate()
            exit(0)


if __name__ == '__main__':
    render_audio_queue = Queue()
    start_audio_cap(None, render_audio_queue)
