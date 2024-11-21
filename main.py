import numpy as np
import pygame
import sys
import json
import os

from macros import *
from objs import *

class Map:
    def __init__(self):
        self.obstacles = [Obstacle() for _ in range(NUM_OBSTACLES)]
        self.danger_map = None  

    def create_obstacles(self):
        [obstacle.randomize() for obstacle in self.obstacles]

    def draw(self, screen):
        [obstacle.draw(screen) for obstacle in self.obstacles]
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, START_POSITION, 10)
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, END_POSITION, 10)

    def generate_danger_map(self):
        self.danger_map = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))

        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                danger = 0
                for obstacle in self.obstacles:
                    ox, oy = obstacle.position
                    radius = obstacle.radius
                    outer_limit = obstacle.radius * 2.5

                    dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                    if dist <= radius:
                        danger = 1
                        break
                    elif dist <= outer_limit:
                        danger += 1 - (np.log10(dist - radius) / np.log10(outer_limit - radius))
                self.danger_map[y, x] = min(danger, 1)

        return self.danger_map
    def save_terrain(self, file_name):
        with open(file_name, 'w') as file:
            file.write(f"obstacles {len(self.obstacles)}\n")
            for obstacle in self.obstacles:
                x, y = obstacle.position
                r = obstacle.radius
                file.write(f"{x} {y} {r}\n")
            print(f"Terrain saved to {file_name}")    
    def load_terrain(self, file_name):
        if not os.path.exists(file_name):
            print(f"{file_name} does not exist. Please ensure the file is available.")
            return

        with open(file_name, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:  
                    break
                if "obstacles" in line:
                    _, num_obstacles = line.split()
                    num_obstacles = int(num_obstacles)
                    self.obstacles = []  
                    
                    for _ in range(num_obstacles):
                        line = file.readline().strip()
                        if line:  
                            x, y, r = [float(value) for value in line.split()]
                            self.obstacles.append(
                                    Obstacle(position=np.array([x, y]), radius=r)
                            )                    

            print(f"Terrain loaded from {file_name}")

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption('Path planning and Trajectory Tracking using Bezier Curve, Genetic Algorithm, Artificial Potential Field and PID Control')
    clock = pygame.time.Clock()

    car = Car("car.png")
    map = Map()
    map.create_obstacles()
    map.save_terrain("terrain.txt")
    map.load_terrain("terrain.txt")
    
    for obstacle in map.obstacles:
        print(f"Position: {obstacle.position}, Radius: {obstacle.radius}")

    model = Genetic_model(map)

    model.generate_initial_population()
#    model.select_elites()
#    model.crossover()
    
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
