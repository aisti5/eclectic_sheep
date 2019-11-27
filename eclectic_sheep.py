from multiprocessing import Process, Queue
from threading import Thread

from audio_capture import start_audio_cap
from ganeretor import start_gan
from shader import start_renderer

if __name__ == '__main__':
    gan_audio_queue = Queue(maxsize=1)
    render_audio_queue = Queue(maxsize=1)
    img_queue = Queue(maxsize=1)
    audio_process = Process(target=start_audio_cap, args=[gan_audio_queue, render_audio_queue])
    gan_process = Process(target=start_gan, args=[gan_audio_queue, img_queue], kwargs={'plot_img': False})
    render_process = Process(target=start_renderer, args=[img_queue, render_audio_queue])

    try:
        audio_process.start()
        gan_process.start()
        render_process.start()
    except KeyboardInterrupt:
        render_process.terminate()
        audio_process.terminate()
        gan_process.terminate()
