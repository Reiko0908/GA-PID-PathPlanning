import numpy as np
import pygame
import math

from macros import *
from bezier import *

CAR_VELOCITY = 200
CAR_ANGULAR_VELOCITY = 2*np.pi/150
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
        self.heading = np.array([1, 0])
        self.position = np.array(START_POSITION)
        self.angular_velocity = 0.0

    def update(self, omega):
        self.heading = rotate_vect(self.heading, omega)
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

    # def angle_to_heading(self):
    #     angle = np.rad2deg(np.arccos(NORTH @ self.heading))
    #     self.heading = np.array([np.sin(np.deg2rad(angle)), -np.cos(np.deg2rad(angle))])
    #     self.heading = self.heading / np.linalg.norm(self.heading)
    
    # def limit_to_screen(self):
    #     self.position[0] = max(0, min(SCREEN_WIDTH, self.position[0]))
    #     self.position[1] = max(0, min(SCREEN_HEIGHT, self.position[1]))
