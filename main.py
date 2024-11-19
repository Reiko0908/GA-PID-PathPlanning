import numpy as np
import pygame
import sys

from macros import *
from objs import *


class Map:
    def __init__(self):
        self.obstacles = [Obstacle() for _ in range(NUM_OBSTACLES)]

    def create_obstacles(self):
        [obstacle.randomize() for obstacle in self.obstacles]

    def draw(self, screen):
        [obstacle.draw(screen) for obstacle in self.obstacles]
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, START_POSITION, 10)
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, END_POSITION, 10)

    def generate_danger_map(self):
        return
    

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
    clock = pygame.time.Clock()

    car = Car("car.png")
    map = Map()
    model = Genetic_model()

    map.create_obstacles()
    model.generate_initial_chromosomes()

    print('Starting Simulation')
    while True:
        clock.tick(SCREEN_FPS)
        screen.fill("gray")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        keys = pygame.key.get_pressed()

        car.update(keys)

        # Draw
        map.draw(screen)
        car.draw(screen)

        pygame.display.flip()

pygame.quit()
