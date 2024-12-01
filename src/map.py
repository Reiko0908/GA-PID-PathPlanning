import matplotlib.pyplot as plt
import numpy as np
import pygame
import math
import os

from macros import *

class Obstacle:
    def __init__(self, position=None, radius=None):
        self.position = np.array(position) if position is not None else np.array([0, 0])
        self.radius = radius if radius is not None else 0

    def randomize(self):
        self.position = np.random.randint([0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.radius = np.random.randint(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)

    def game_draw(self, screen):
        pygame.draw.circle(screen, OBSTACLE_COLOR, self.position.tolist(), self.radius)
    
class Map:
    def __init__(self):
        self.obstacles = [Obstacle() for _ in range(NUM_OBSTACLES)]
        self.danger_map = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))

    def create_obstacles(self):
        print("Create obstacles")
        [obstacle.randomize() for obstacle in self.obstacles]

    def game_draw(self, screen):
        [obstacle.game_draw(screen) for obstacle in self.obstacles]
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, START_POSITION, 10)
        pygame.draw.circle(screen, MAP_END_POINTS_COLOR, END_POSITION, 10)

    def generate_danger_map(self):
        print("Generating danger map")
        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                danger = 0
                for obstacle in self.obstacles:
                    ox, oy = obstacle.position
                    radius = obstacle.radius
                    outer_limit = obstacle.radius + 100

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

    def save_danger_map(self, file_name):
        with open(file_name, 'w') as file:
            file.write(f"{SCREEN_WIDTH} {SCREEN_HEIGHT}\n") 
            for y in range(SCREEN_HEIGHT):
                for x in range(SCREEN_WIDTH):
                    danger = self.danger_map[y, x]
                    file.write(f"{x} {y} {danger:.5f}\n")
        print(f"Danger map saved to {file_name}")

    def load_danger_map(self, file_name):
        if not os.path.exists(file_name):
            print(f"{file_name} does not exist. Please ensure the file is available.")
            return
        
        with open(file_name, 'r') as file:
            dimensions = file.readline().strip().split()
            width, height = map(int, dimensions)
            if width != SCREEN_WIDTH or height != SCREEN_HEIGHT:
                print("Error: Loaded danger map dimensions do not match the screen size.")
                return
            
            self.danger_map = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
            for line in file:
                x, y, danger = line.strip().split()
                x, y = int(x), int(y)
                danger = float(danger)
                self.danger_map[y, x] = danger
        
        print(f"Danger map loaded from {file_name}")

    def plot_3d_terrain(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, SCREEN_WIDTH, SCREEN_WIDTH)
        y = np.linspace(0, SCREEN_HEIGHT, SCREEN_HEIGHT)
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, self.danger_map, cmap='hot', edgecolor='none')
        fig.colorbar(surf, ax=ax, label='Danger Level')
        ax.set_title("3D Danger Map")
        ax.set_xlabel('Width (X)')
        ax.set_ylabel('Height (Y)')
        ax.set_zlabel('Danger Level')
        plt.show()    
