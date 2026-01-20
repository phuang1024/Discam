"""
PTZ control thread.
"""

import pygame
import time

import cv2
import numpy as np

from constants import *

pygame.init()


class GUIController:
    """
    PTZ controller and visualization via pygame.
    """

    def __init__(self):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))

    def update(self, state, curr_pos, dt):
        """
        Draw frame on display, and return new pos.
        """
        pygame.event.get()
        self.draw_frame(state.frameq[-1])
        pygame.display.update()

        ctrl = np.array(self.compute_control())
        new_pos = curr_pos + ctrl * dt
        return new_pos

    def draw_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.window.blit(frame, (0, 0))

    def compute_control(self):
        mouse_press = pygame.mouse.get_pressed()
        if not mouse_press[0]:
            return (0, 0, 0)

        width, height = self.window.get_size()
        mx, my = pygame.mouse.get_pos()

        max_speed = GUI_PT_SPEED * 3600
        pan = np.interp(mx, [0, width], [-max_speed, max_speed])
        tilt = -1 * np.interp(my, [0, height], [-max_speed, max_speed])

        return (pan, tilt, 0)


def control_thread(state: ThreadState):
    gui = GUIController()

    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)

    last_time = time.time()
    while state.run:
        time.sleep(1 / FPS)
        if len(state.frameq) == 0:
            continue

        dt = time.time() - last_time
        last_time = time.time()

        curr_pos = gui.update(state, curr_pos, dt)

        state.camera.set(cv2.CAP_PROP_PAN, curr_pos[0])
        state.camera.set(cv2.CAP_PROP_TILT, curr_pos[1])
        #state.camera.set(cv2.CAP_PROP_ZOOM, curr_pos[2])
