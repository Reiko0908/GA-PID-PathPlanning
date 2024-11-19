import numpy as np
import pygame
import sys

from macros import *
from objs import *

NUM_OBSTACLES = 20
MAX_OBSTACLE_SIZE = 100
MIN_OBSTACLE_SIZE = 20
OBSTACLE_COLOR = (150, 150, 150)

class Obstacle:
    def __init__(self):
        self.position = np.array([])
        self.radius = np.array([])

    def randomize(self):
        self.position = np.random.randint([0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.radius = np.random.randint(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)

    def draw(self, screen):
        pygame.draw.circle(screen, OBSTACLE_COLOR, self.position.tolist(), self.radius)
    

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
    clock = pygame.time.Clock()

    car = Car()
    bezier_path = Bezier()
    obstacles = [Obstacle() for i in range(NUM_OBSTACLES)]

    bezier_path.randomize()
    car.randomize()
    [obs.randomize() for obs in obstacles]

    robot_img = pygame.image.load("robot.png")
    robot_img.convert()
    robot = robot_img.get_rect()
    print(robot.center)

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
        [obs.draw(screen) for obs in obstacles]
        car.draw(screen)

        pygame.display.flip()

pygame.quit()
