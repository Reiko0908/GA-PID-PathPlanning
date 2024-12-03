import numpy as np
import pygame
import math

from macros import *
from bezier import *

CAR_SCALE = 0.15

NORTH = np.array([0, -1])

def rotate_vect(vector, angle_radian):
    rotation_matrix = np.array([
            [np.cos(angle_radian), -np.sin(angle_radian)],
            [np.sin(angle_radian), np.cos(angle_radian)]
            ])
    return rotation_matrix @ vector

class Car:
    def __init__(self, image_path):
        image = pygame.image.load(image_path)
        self.img = pygame.transform.scale_by(image, CAR_SCALE)
        self.heading = np.array([0, 1])
        self.position = np.array(START_POSITION)
        self.angular_velocity = 0.0
        self.time = 0

    def update(self, omega):
        self.heading = rotate_vect(self.heading, omega)
        print(omega)
        self.position = self.position + self.heading * CAR_VELOCITY / SCREEN_FPS
        self.heading = self.heading /np.linalg.norm(self.heading)


    def game_draw(self, screen):
        angle = np.rad2deg(np.arccos(NORTH @ self.heading))
        if (np.cross(NORTH, self.heading) > 0):
            image  = pygame.transform.rotate(self.img, -angle)
        else:
            image  = pygame.transform.rotate(self.img, angle)
        rect = image.get_rect(center=self.position.tolist())
        screen.blit(image, rect)
        self.time += 1

    def noise(self, frequency : int):
        if frequency == 0:
            return
        elif self.time % (SCREEN_FPS / frequency) == 0:
            self.noise_magnitude = np.random.normal(0, np.sqrt(0.035))
            self.noise_angle = np.deg2rad(np.random.randint(-45, 45))
            self.position = self.position + self.noise_magnitude*(rotate_vect(self.heading, self.noise_angle))
