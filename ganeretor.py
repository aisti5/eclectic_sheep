import time
from multiprocessing import Queue

import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, utils

from audio_gan import _plot_output


def load_gan(size=256):
    return BigGAN.from_pretrained('biggan-deep-{}'.format(size))


def load_inputs(class_vector=None, noise_vector=None):
    # Prepare a input
    if class_vector is not None:
        class_vector_cpu = torch.from_numpy(class_vector).float()
        class_vector = class_vector_cpu.to('cuda')
    if noise_vector is not None:
        noise_vector_cpu = torch.from_numpy(noise_vector).float()
        noise_vector = noise_vector_cpu.to('cuda')

    return class_vector, noise_vector


def start_gan(audio_queue: Queue, img_queue: Queue, plot_img=False):
    biggan = load_gan(512)
    biggan.to('cuda')
    old_img = None

    noise_vector = truncated_noise_sample(truncation=.2, batch_size=1)
    _, noise_vector = load_inputs(None, noise_vector)

    with torch.no_grad():
        while True:
            try:
                start_time = time.time()
                if not audio_queue.empty():
                    class_vector_cpu = audio_queue.get()
                else:
                    continue
                class_vector, _ = load_inputs(class_vector_cpu, None)
                output = biggan(noise_vector, class_vector, .2)  # type: torch.Tensor
                output = output.to('cpu')
                old_img, pil_img = _plot_output(output, class_vector_cpu, old_img, plot_img)
                if not img_queue.full():
                    img_queue.put(old_img, False)
                print("BigGAN Refresh rate: {:.2f}Hz".format(1 / (time.time() - start_time)))
            except KeyboardInterrupt:
                exit(0)
