import numpy as np
import pygame
import math

from macros import *

CAR_CORLOR = (255, 0, 0)
CAR_VELOCITY = 150
CAR_ANGULAR_VELOCITY = 2*np.pi/150

BEZIER_ORDER = 3
BEZIER_RESOLUTION = 50
BEZIER_LOCAL_POINT_COLOR = (255, 255, 0)
BEZIER_CONTROL_POINT_COLOR = (0, 0, 255)

NUM_OBSTACLES = 20
MAX_OBSTACLE_SIZE = 100
MIN_OBSTACLE_SIZE = 20
OBSTACLE_COLOR = (150, 150, 150)

def rotate_vect(vector, angle_radian):
    rotation_matrix = np.array([
            [np.cos(angle_radian), -np.sin(angle_radian)],
            [np.sin(angle_radian), np.cos(angle_radian)]
            ])
    return rotation_matrix @ vector

class Car:
    def __init__(self, image_path):
        image = pygame.image.load(image_path)
        self.img = pygame.transform.scale_by(image,0.2)
        self.heading = np.array([0, -1])
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

# -------------------------------------------------------------------------------------------------------

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

class Obstacle:
    def __init__(self):
        self.position = np.array([])
        self.radius = np.array([])

    def randomize(self):
        self.position = np.random.randint([0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.radius = np.random.randint(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)

    def draw(self, screen):
        pygame.draw.circle(screen, OBSTACLE_COLOR, self.position.tolist(), self.radius)

# -------------------------------------------------------------------------------------------------------
PATH_LENGTH_PRIORITIZE_FACTOR = 0.8
PATH_SAFETY_PRIORITIZE_FACTOR = 1 - PATH_LENGTH_PRIORITIZE_FACTOR

def measure_bezier_length(bezier):
    return
def measure_bezier_danger(bezier):
    return

class Genetic_model:
    def __init__(self):
        self.chromosomes = np.array([])
        return

    def fitness_function(self, chromosome):
        return

    def evaluate_population(self):
        return
