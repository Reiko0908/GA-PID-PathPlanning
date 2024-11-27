import numpy as np
from numpy.random import f
import pygame
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from macros import *
from bezier import *
from map import *
from genetic_model import *

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
    # map.save_terrain("terrain.txt")
    map.load_terrain("terrain.txt")
    # map.generate_danger_map()
    # map.save_danger_map("danger_map.txt")
    map.load_danger_map("danger_map.txt")
   
    FITNESS_THRESHOLD = 0.001  # Stop if fitness reaches below 0.2
    best_fitness_so_far = float('inf')  # Initialize with infinity

for generation in range(NUM_GENERATIONS):
    model = Genetic_model()
    model.generate_initial_population()
    print(f"Generation {generation + 1}")

    # Evaluate population and calculate fitness
    model.evaluate_population(map)  # Pass the map only, population is handled inside the model

    # Get the best fitness score in the current generation
    current_best_fitness = min(model.fitness_scores)
    print(f"Best fitness in this generation: {current_best_fitness}")

    # Update the best fitness so far if current generation has a better fitness
    if current_best_fitness < best_fitness_so_far:
        best_fitness_so_far = current_best_fitness

    # Stop if the fitness threshold is reached
    if current_best_fitness <= FITNESS_THRESHOLD:
        print("Optimal solution found!")
        model.validate(map)  # Validate the best chromosome
        print(len(model.chromosomes))
        plot_terrain(map)  # Plot the terrain
        plot_bezier(chromosome_to_bezier(model.chromosomes[0]))  # Plot the Bezier curve of the best chromosome
        plt.show()  # Show the plots
        break  # Exit the loop

    # Perform genetic operations (selection, crossover, mutation)
    model.select_elites()
    model.crossover()
    model.mutate() 
    # Optionally print the best fitness so far at the end of each generation
    print(f"Best fitness so far: {best_fitness_so_far}")
