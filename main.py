import numpy as np
import pygame
import sys

from macros import *
from objs import *


class Map:
    def __init__(self):
        self.obstacles = [Obstacle() for _ in range(NUM_OBSTACLES)]

    def randomize(self):
        [obstacle.randomize() for obstacle in self.obstacles]

    def draw(self, screen):
        [obstacle.draw(screen) for obstacle in self.obstacles]


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
    clock = pygame.time.Clock()

    car = Car("car.png")
    bezier_path = Bezier()
    map = Map()

    bezier_path.randomize()
    car.randomize()
    map.randomize()

    while True:
        clock.tick(SCREEN_FPS)
        screen.fill("gray")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        keys = pygame.key.get_pressed()

        car.update(keys)

        # Draw
        bezier_path.draw(screen)
        map.draw(screen)
        car.draw(screen)

        pygame.display.flip()

pygame.quit()
