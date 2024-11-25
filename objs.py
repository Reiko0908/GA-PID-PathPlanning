import numpy as np
import pygame
import math

from macros import *

CAR_VELOCITY = 200
CAR_ANGULAR_VELOCITY = 2*np.pi/150

def rotate_vect(vector, angle_radian):
    rotation_matrix = np.array([
            [np.cos(angle_radian), -np.sin(angle_radian)],
            [np.sin(angle_radian), np.cos(angle_radian)]
            ])
    return rotation_matrix @ vector

class Car:
    def __init__(self, image_path):
        image = pygame.image.load(image_path)
        self.img = pygame.transform.scale_by(image, 0.15)
        self.heading = np.array([0, -1])
        self.position = np.array(START_POSITION)

    def update(self, key):
        if key[pygame.K_d]:
            self.heading = rotate_vect(self.heading, CAR_ANGULAR_VELOCITY)
        if key[pygame.K_a]:
            self.heading = rotate_vect(self.heading, -CAR_ANGULAR_VELOCITY)
        if key[pygame.K_w]:
            self.position = self.position + CAR_VELOCITY * self.heading / SCREEN_FPS
        if key[pygame.K_s]:
            self.position = self.position - CAR_VELOCITY * self.heading / SCREEN_FPS
        return

    def draw(self, screen):
        self.heading = self.heading /np.linalg.norm(self.heading)
        north = np.array([0, -1])
        angle = np.rad2deg(np.arccos(north @ self.heading))
        if (np.cross(north, self.heading) > 0):
            image  = pygame.transform.rotate(self.img, -angle)
        else:
            image  = pygame.transform.rotate(self.img, angle)
        rect = image.get_rect(center=self.position.tolist())
        screen.blit(image, rect)
        return

