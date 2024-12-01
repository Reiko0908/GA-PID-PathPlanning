import numpy as np
import pygame
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from map import *
from car import *
from genetic_model import *
from bezier import *
from macros import *

KP = 0.005
KI = 0.000001
KD = 0.00003

def pid(car, bezier):
    if not hasattr(pid, 'integral_term'):
        pid.integral_term = 0
        pid.prev_y_err = 0

    # z is car's predicted next position
    # p is the projection of z on the bezier curve
    # y_err = distance between z and p
    # omega is the angular velocity of the car

    z = car.position + car.heading * CAR_VELOCITY / SCREEN_FPS
    p, dist_zp = bezier.get_projection_of(z)
    vec_zp = p - z
    y_err = np.cross(car.heading, vec_zp) * dist_zp
    
    p_term = KP * y_err / SCREEN_FPS
    pid.integral_term += KI * (y_err + pid.prev_y_err) / 2.0 / SCREEN_FPS
    d_term = KD * (y_err - pid.prev_y_err) * SCREEN_FPS

    omega = p_term + pid.integral_term + d_term
    pid.prev_y_err = y_err

    return omega


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
    clock = pygame.time.Clock()

    car = Car("car.png")
    map = Map()
    # map.create_obstacles()
    # map.save_terrain("terrain.txt")
    map.load_terrain("terrain.txt")
    map.load_danger_map("danger_map.txt")

    model = Genetic_model()
    model.load_best_chromosomes("best_chromosome.txt")

    path = chromosome_to_bezier(model.best_chromosome)

    omega = 0
    while True:
        clock.tick(SCREEN_FPS)
        screen.fill("gray")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_f]: continue

        car.update(omega)
        omega = pid(car, path)

        map.game_draw(screen)
        car.game_draw(screen)
        car.noise(60)
        path.game_draw(screen)
        pygame.display.flip()

pygame.quit()
