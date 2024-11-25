import numpy as np
import pygame
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from macros import *

# if __name__ == "__main__":
#     pygame.init()
#     screen = pygame.display.set_mode((1500, 800))
#     pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
#     clock = pygame.time.Clock()
#
#     car = Car("car.png")
#     map = Map()
#     map.create_obstacles()
#     map.save_terrain("terrain.txt")
#     map.load_terrain("terrain.txt")
#
#     map.generate_danger_map()
#     danger_map = map.danger_map
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Generate coordinate grid
#     x = np.linspace(0, SCREEN_WIDTH, SCREEN_WIDTH)
#     y = np.linspace(0, SCREEN_HEIGHT, SCREEN_HEIGHT)
#     x, y = np.meshgrid(x, y)
#
#     # Plotting the surface
#     surf = ax.plot_surface(x, y, danger_map, cmap='hot', edgecolor='none')
#
#     # Adding color bar
#     fig.colorbar(surf, ax=ax, label='Danger Level')
#
#     # Set labels and title
#     ax.set_title("3D Danger Map")
#     ax.set_xlabel('Width (X)')
#     ax.set_ylabel('Height (Y)')
#     ax.set_zlabel('Danger Level')
#
#     # Show the plot
#     plt.show()    
#
#     model = Genetic_model(map)
#
#     for obstacle in map.obstacles:
#         print(f"Position: {obstacle.position}, Radius: {obstacle.radius}")
#     model.generate_initial_population()
# #    model.select_elites()
# #    model.crossover()    
#     print('Starting Simulation')
#     while True:
#         clock.tick(SCREEN_FPS)
#         screen.fill("gray")
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 sys.exit()
#
#         keys = pygame.key.get_pressed()
#
#         car.update(keys)
#
#         # Draw
#         map.draw(screen)
#         car.draw(screen)
#
#         pygame.display.flip()
#
# pygame.quit()
