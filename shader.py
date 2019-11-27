import copy
import time
import os
from multiprocessing import Queue

import numpy as np
from glumpy import app, gl, glm, gloo, data


def read_file(path):
    with open(path, 'r') as fp:
        return fp.read()


def cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal', np.float32, 3)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6, 1,
               1, 6, 7, 2, 7, 4, 3, 2, 4, 7, 6, 5]
    faces_n = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
               3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    faces_t = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
               3, 2, 1, 0, 0, 1, 2, 3, 0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal'] = n[faces_n]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
        np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)

    return vertices, filled


phi, theta, new_img = 40, 30, None
tes1 = data.load('tes.jpg')
tes2 = data.load('tes2.jpg')
flipper = 0
new_img = None
old_img = None

shared_img_queue = None
shared_audio_queue = None

i_time = 0


def start_renderer(img_queue: Queue, audio_queue: Queue):
    global new_img, tes1, tes2, shared_img_queue, shared_audio_queue

    vertices, indices = cube()
    shared_img_queue = img_queue
    shared_audio_queue = audio_queue
    vertex = read_file(os.path.join('shaders', 'triangle.vert'))
    fragment = read_file(os.path.join('shaders', 'triangle.frag'))
    shader_program = gloo.Program(vertex, fragment)
    shader_program.bind(vertices)
    new_img = img_queue.get() if img_queue is not None else tes1
    shader_program['u_texture_from'] = tes1
    shader_program['u_texture_to'] = tes2
    shader_program['u_model'] = np.eye(4, dtype=np.float32)
    shader_program['u_view'] = glm.translation(0, 0, -5)
    shader_program['time'] = time.time()

    window = app.Window(width=1200, height=720, title="Eclectic Sheep",
                        color=(0.0, 0.0, 0.0, 1.0))

    @window.event
    def on_draw(*args):
        global phi, theta, new_img, shared_img_queue, shared_audio_queue, i_time, old_img
        i_time += .05

        window.clear()
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        shader_program.draw(gl.GL_TRIANGLES, indices)
        shader_program['time'] = i_time
        try:
            old_img = new_img
            new_img = shared_img_queue.get(False)  # type: np.ndarray
            shader_program['u_texture_from'] = old_img
            shader_program['u_texture_to'] = new_img
        except Exception as ex:
            pass

        # Rotate cube
        beat_rotation_factor = 0
        mids_rotation_factor = 0
        try:
            audio_data = shared_audio_queue.get(False)  # type: np.ndarray
            beat_rotation_factor = 1e-3 * np.mean(audio_data[0:300])
            mids_rotation_factor = 1e-3 * np.mean(audio_data[300:600])
            # print(beat_rotation_factor, mids_rotation_factor)
        except Exception as ex:
            # print(ex)
            pass
        theta += beat_rotation_factor + .10  # degrees
        phi += mids_rotation_factor + .10  # degrees
        model = np.eye(4, dtype=np.float32)
        glm.rotate(model, theta, 0, 0, 1)
        glm.rotate(model, phi, 0, 1, 0)
        shader_program['u_model'] = model

    @window.event
    def on_resize(width, height):
        shader_program['u_projection'] = glm.perspective(45.0, width / float(height), 2.0, 100.0)

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    try:
        app.run(framerate=120)
    except AttributeError:
        print("Exiting")


if __name__ == '__main__':
    start_renderer(None)
