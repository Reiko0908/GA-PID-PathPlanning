import numpy as np
import pygame
import math

from macros import *
from bezier import *

CAR_VELOCITY = 200
CAR_ANGULAR_VELOCITY = 2*np.pi/150
CAR_SCALE = 0.15

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
        self.heading = np.array([0, -1])
        self.position = np.array(START_POSITION)

    def game_draw(self, screen):
        self.heading = self.heading /np.linalg.norm(self.heading)
        self.north = np.array([0, -1])
        self.angle = np.rad2deg(np.arccos(self.north @ self.heading))
        if (np.cross(self.north, self.heading) > 0):
            image  = pygame.transform.rotate(self.img, -self.angle)
        else:
            image  = pygame.transform.rotate(self.img, self.angle)
        rect = image.get_rect(center=self.position.tolist())
        screen.blit(image, rect)

    def angle_to_heading(self):
        self.heading = np.array([np.sin(np.deg2rad(self.angle)), -np.cos(np.deg2rad(self.angle))])
        self.heading = self.heading / np.linalg.norm(self.heading)
    
    def limit_to_screen(self):
        self.position[0] = max(0, min(SCREEN_WIDTH, self.position[0]))
        self.position[1] = max(0, min(SCREEN_HEIGHT, self.position[1]))
 
# if __name__ == "__main__":
#     pygame.init()
#     screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
#     pygame.display.set_caption("Car test")
#     running = True
#     car = Car('car.png')
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#         screen.fill("Black")
#         keys = pygame.key.get_pressed()
#         # car.update(keys)
#         car.game_draw(screen)
#         pygame.time.Clock().tick(SCREEN_FPS)
#         pygame.display.flip()
#     pygame.quit()
