import numpy as np
from numpy.random import f
import pygame
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D

import bezier
from macros import *
from bezier import *
from map import *
from genetic_model import *

def plot_3d_terrain(map):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, SCREEN_WIDTH, SCREEN_WIDTH)
    y = np.linspace(0, SCREEN_HEIGHT, SCREEN_HEIGHT)
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, map.danger_map, cmap='hot', edgecolor='none')
    fig.colorbar(surf, ax=ax, label='Danger Level')
    ax.set_title("3D Danger Map")
    ax.set_xlabel('Width (X)')
    ax.set_ylabel('Height (Y)')
    ax.set_zlabel('Danger Level')
    plt.show()    

def plot_terrain(map):
    plt.figure(figsize=(SCREEN_WIDTH/100, SCREEN_HEIGHT/100))
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)

    obstacle_color = (OBSTACLE_COLOR[0]/255, OBSTACLE_COLOR[1]/255, OBSTACLE_COLOR[2]/255)
    for obs in map.obstacles:
        plt.scatter(obs.position[0], obs.position[1], s=obs.radius**2, color=obstacle_color)

    map_end_points_color = (MAP_END_POINTS_COLOR[0]/255, MAP_END_POINTS_COLOR[1]/255, MAP_END_POINTS_COLOR[2]/255)
    plt.scatter(START_POSITION[0], START_POSITION[1], color=map_end_points_color)
    plt.scatter(END_POSITION[0], END_POSITION[1], color=map_end_points_color)

def plot_bezier(bezier):
    t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
    local_points = np.zeros((BEZIER_RESOLUTION, 2))
    bezier_local_point_color = (BEZIER_LOCAL_POINT_COLOR[0]/255, BEZIER_LOCAL_POINT_COLOR[1]/255, BEZIER_LOCAL_POINT_COLOR[2]/255)

    bezier_order = len(bezier.control_points.tolist()) - 1
    for i in range(BEZIER_RESOLUTION):
        for j in range(bezier_order + 1):
            t = t_values[i]
            temp = math.comb(bezier_order, j) * t**j * (1-t)**(bezier_order-j)
            local_points[i] = local_points[i] + bezier.control_points[j] * temp
    
    plt.plot(local_points[:, 0], local_points[:, 1], color=bezier_local_point_color)

if __name__ == "__main__":
    map = Map()
    # map.create_obstacles()
    # map.save_terrain("../data/terrain.txt")
    map.load_terrain("../data/terrain.txt")
    # map.generate_danger_map()
    # map.save_danger_map("../data/danger_map.txt")
    map.load_danger_map("../data/danger_map.txt")
    model = Genetic_model()
    model.generate_initial_population() 

    plot_terrain(map)

    best_fitness = 1.0
    best_chromosome = None

    for epoch_num in range(NUM_EPOCH):
        model.evaluate_population(map)
        model.separate_elites()
        print(len(model.elite_indices))
        model.save_epoch_results()
        model.track_best_chromosome()
        model.crossover()
        model.mutate()

    model.save_best_chromosome("../data/best_chromosome.txt")

    plot_bezier(chromosome_to_bezier(model.best_chromosome))
    # plt.plot(range(NUM_EPOCH), model.saved_data)
    # plt.savefig("train_results.png")
    plt.show()
