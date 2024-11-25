import numpy as np
import pygame
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from macros import *
from bezier import *
from map import *
from genetic_model import *

def plot_terrain(map):
    plt.figure(figsize=(SCREEN_WIDTH/100, SCREEN_HEIGHT/100))
    obstacle_color = (OBSTACLE_COLOR[0]/255, OBSTACLE_COLOR[1]/255, OBSTACLE_COLOR[2]/255)
    for obs in map.obstacles:
        plt.scatter(obs.position[0], obs.position[1], s=obs.radius**2, color=obstacle_color)

    map_end_points_color = (MAP_END_POINTS_COLOR[0]/255, MAP_END_POINTS_COLOR[1]/255, MAP_END_POINTS_COLOR[2]/255)
    plt.scatter(START_POSITION[0], START_POSITION[1], color=map_end_points_color)
    plt.scatter(END_POSITION[0], END_POSITION[1], color=map_end_points_color)

    plt.show()

def plot_bezier_with_terrain(map, bezier):
    # Plot the terrain
    plt.figure(figsize=(SCREEN_WIDTH/100, SCREEN_HEIGHT/100))
    obstacle_color = (OBSTACLE_COLOR[0]/255, OBSTACLE_COLOR[1]/255, OBSTACLE_COLOR[2]/255)
    for obs in map.obstacles:
        plt.scatter(obs.position[0], obs.position[1], s=obs.radius**2, color=obstacle_color)

    map_end_points_color = (MAP_END_POINTS_COLOR[0]/255, MAP_END_POINTS_COLOR[1]/255, MAP_END_POINTS_COLOR[2]/255)
    plt.scatter(START_POSITION[0], START_POSITION[1], color=map_end_points_color)
    plt.scatter(END_POSITION[0], END_POSITION[1], color=map_end_points_color)

    # Plot the bezier
    t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
    local_points = np.zeros((BEZIER_RESOLUTION, 2))
    bezier_local_point_color = (BEZIER_LOCAL_POINT_COLOR[0]/255, BEZIER_LOCAL_POINT_COLOR[1]/255, BEZIER_LOCAL_POINT_COLOR[2]/255)

    for i in range(BEZIER_RESOLUTION):
        for j in range(BEZIER_ORDER + 1):
            t = t_values[i]
            temp = math.comb(BEZIER_ORDER, j) * t**j * (1-t)**(BEZIER_ORDER-j)
            local_points[i] = local_points[i] + bezier.control_points[j] * temp
    
    plt.plot(local_points[:, 0], local_points[:, 1], color=bezier_local_point_color)
    plt.show()

if __name__ == "__main__":

    map = Map()
    map.create_obstacles()
    # map.save_terrain("terrain.txt")
    map.load_terrain("terrain.txt")
    map.generate_danger_map()

    model = Genetic_model()
    model.generate_initial_population()
    print(model.evaluate_population(map))


