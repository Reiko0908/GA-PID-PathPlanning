import numpy as np
import pygame
import math

from macros import *

CAR_CORLOR = (255, 0, 0)
CAR_TRIANGLE_BASE = 20
CAR_TRANGLE_HEIGHT = 30
CAR_VELOCITY = 100
CAR_ANGULAR_VELOCITY = 2*np.pi/100

def rotate_vect(vector, angle_radian):
    rotation_matrix = np.array([
            [np.cos(angle_radian), -np.sin(angle_radian)],
            [np.sin(angle_radian), np.cos(angle_radian)]
            ])
    return rotation_matrix @ vector

class Car:
    def __init__(self):
        self.heading = np.array([])
        self.position = np.array([])

    def randomize(self):
        self.position = np.random.randint([0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.heading = rotate_vect(np.array([1, 0]), np.random.rand() * 2* np.pi)

    def update(self, key):
        if key[pygame.K_d]:
            self.heading = rotate_vect(self.heading, CAR_ANGULAR_VELOCITY)
        if key[pygame.K_a]:
            self.heading = rotate_vect(self.heading, -CAR_ANGULAR_VELOCITY)
        if key[pygame.K_w]:
            self.position = self.position + CAR_VELOCITY * (1/60) * self.heading
        if key[pygame.K_s]:
            self.position = self.position - CAR_VELOCITY * (1/60) * self.heading

    def draw(self, screen):
        normal_vect = np.array([-self.heading[1], self.heading[0]])
        points = [
                (self.position + (2/3) * CAR_TRANGLE_HEIGHT * self.heading).tolist(),
                (self.position - (1/3) * CAR_TRANGLE_HEIGHT * self.heading + 0.5 * CAR_TRIANGLE_BASE * normal_vect).tolist(),
                (self.position - (1/3) * CAR_TRANGLE_HEIGHT * self.heading - 0.5 * CAR_TRIANGLE_BASE * normal_vect).tolist()
                ]
        pygame.draw.polygon(screen, CAR_CORLOR, points)

# -------------------------------------------------------------------------------------------------------

BEZIER_ORDER = 3
BEZIER_RESOLUTION = 50
BEZIER_LOCAL_POINT_COLOR = (255, 255, 0)
BEZIER_CONTROL_POINT_COLOR = (0, 0, 255)

class Bezier:
    def __init__(self):
        self.control_points = np.array([])
    
    def randomize(self):
        self.control_points = np.random.rand(BEZIER_ORDER + 1, 2)
        self.control_points[:, 0] = self.control_points[:, 0] * SCREEN_WIDTH
        self.control_points[:, 1] = self.control_points[:, 1] * SCREEN_HEIGHT
        return

    def update(self):
        return

    def draw(self, screen):
        for point in self.control_points:
            pygame.draw.circle(screen, BEZIER_CONTROL_POINT_COLOR, point.tolist(), 10)

        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        for t in t_values:
            local_point = np.array([0, 0])
            for i in range(BEZIER_ORDER + 1):
                local_point = local_point + math.comb(BEZIER_ORDER, i) * t**i * (1-t)**(BEZIER_ORDER-i) * self.control_points[i]
            pygame.draw.circle(screen, BEZIER_LOCAL_POINT_COLOR, local_point.tolist(), 5)

