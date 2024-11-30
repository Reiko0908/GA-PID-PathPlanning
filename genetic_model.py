import matplotlib.pyplot as plt
import numpy as np
import pygame
import math

from macros import *
from bezier import *

CROSSOVER_RATIO = 0.5
ELIMINATION_THRESHOLD = 0.25
MUTATION_RATIO = 0.2
POPULATION = 200
CHROMOSOME_INITIAL_LENGTH = 7
NUM_EPOCH = 100

def chromosome_to_bezier(chromosome):
    bezier = Bezier()
    bezier.control_points= np.array([gene for gene in chromosome])
    return bezier

def fitness_function(chromosome, map):
    bezier = chromosome_to_bezier(chromosome)
    t_values = np.linspace(0, 1, BEZIER_RESOLUTION)

    for obs in map.obstacles:
        _, projection_length = bezier.get_projection_of(obs.position)
        if projection_length < obs.radius:
            return 1.0

    sampled_points = np.array([bezier.calculate_local_point(t) for t in t_values[1:-1]])
    sample_points_as_int = np.floor(sampled_points).astype(int)  # Use floor for safety
    total_danger = 0
    for point in sample_points_as_int:
        x, y = point[0], point[1]
        total_danger += map.danger_map[y, x]

    return total_danger / (BEZIER_RESOLUTION - 2)

class Genetic_model:
    def __init__(self):
        # print("Initializing Model")
        self.chromosomes = []
        self.fitness_scores = [0] * POPULATION
        self.elite_indices = []
        self.non_elite_indices = []
        self.saved_data = []
        self.best_finess = 1.0
        self.best_chromosome = []

    def generate_initial_population(self):
        for _ in range(POPULATION):
            chromo = [START_POSITION]
            for __ in range(CHROMOSOME_INITIAL_LENGTH - 2):
                while(True):
                    gene = [np.random.randint(SCREEN_WIDTH), np.random.randint(SCREEN_HEIGHT)]
                    if gene != START_POSITION and gene != END_POSITION: break
                chromo.append(gene)
            chromo.append(END_POSITION)
            self.chromosomes.append(chromo)

    def evaluate_population(self, map):
        for i in range(POPULATION):
            self.fitness_scores[i] = fitness_function(self.chromosomes[i], map)

    def separate_elites(self):
        sorted_indices = np.argsort(self.fitness_scores)  
        # self.saved_data.append(self.fitness_scores[sorted_indices[0]])
        i = 0
        while(self.fitness_scores[sorted_indices[i]] < ELIMINATION_THRESHOLD):
            i += 1
        self.elite_indices = sorted_indices[:i]
        self.non_elite_indices = sorted_indices[i:] 

    def crossover(self):
        # Crossover the elites
        num_crossover = int(len(self.elite_indices) * CROSSOVER_RATIO)
        if num_crossover % 2 != 0:
            num_crossover -=1
        chosen_parent_indices = np.random.choice(
                len(self.elite_indices), num_crossover, replace=False
                )
        
        np.random.shuffle(chosen_parent_indices)
        for i in range(0,len(chosen_parent_indices)-1, 2):
            mom = self.chromosomes[chosen_parent_indices[i]]
            dad = self.chromosomes[chosen_parent_indices[i+1]]
            min_length = min(len(mom), len(dad)) # perform randomize with the smaller gene length
            cross_over_point = np.random.randint(1, min_length - 1) # ignore start and end genes
            son = mom[:cross_over_point] + dad[cross_over_point:]
            daughter = dad[:cross_over_point] + mom[cross_over_point:]
            # overwrite mom and dad with 2 children
            self.chromosomes[chosen_parent_indices[i]] = son
            self.chromosomes[chosen_parent_indices[i+1]] = daughter

        # Crossover the non-elites
        num_crossover = int(len(self.non_elite_indices) * CROSSOVER_RATIO)
        if num_crossover % 2 != 0:
            num_crossover -=1
        chosen_parent_indices = np.random.choice(
                len(self.non_elite_indices), num_crossover, replace=False
                )
        
        np.random.shuffle(chosen_parent_indices)
        for i in range(0,len(chosen_parent_indices)-1, 2):
            mom = self.chromosomes[chosen_parent_indices[i]]
            dad = self.chromosomes[chosen_parent_indices[i+1]]
            min_length = min(len(mom), len(dad)) # perform randomize with the smaller gene length
            cross_over_point = np.random.randint(1, min_length - 1) # ignore start and end genes
            son = mom[:cross_over_point] + dad[cross_over_point:]
            daughter = dad[:cross_over_point] + mom[cross_over_point:]
            # overwrite mom and dad with 2 children
            self.chromosomes[chosen_parent_indices[i]] = son
            self.chromosomes[chosen_parent_indices[i+1]] = daughter

    def mutate_add_gene(self, chromo_index):
       # Access the actual chromosome using the index from non_elite_indices
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1)
        new_gene = [np.random.randint(SCREEN_WIDTH), np.random.randint(SCREEN_HEIGHT)]
        self.chromosomes[chromo_index].insert(position, new_gene)

    def mutate_remove_gene(self, chromo_index):
        if(len(self.chromosomes[chromo_index])) <= 3:
            return
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        del self.chromosomes[chromo_index][position]

    def mutate_edit_gene(self, chromo_index):
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        self.chromosomes[chromo_index][position] = [
                np.random.randint(SCREEN_WIDTH),
                np.random.randint(SCREEN_HEIGHT)
            ]

    def mutate(self):
        mutate_chosen = np.random.choice([True, False], size=len(self.non_elite_indices), p=[MUTATION_RATIO, 1 - MUTATION_RATIO])
        for i, mutate_flag in enumerate(mutate_chosen):
            if mutate_flag:
                chromo_index = self.non_elite_indices[i]  # This is now an integer, not a list
                mutation_type = np.random.choice(['add', 'remove', 'edit'])
                if mutation_type == 'add':
                    self.mutate_add_gene(chromo_index)
                elif mutation_type == 'remove':
                    self.mutate_remove_gene(chromo_index)
                elif mutation_type == 'edit':
                    self.mutate_edit_gene(chromo_index)

    # def full_fill_population(self):
    #     valid_chromosomes = len(self.chromosomes)
    #     while valid_chromosomes < POPULATION:
    #         chromo = [START_POSITION]
    #         for _ in range(CHROMOSOME_INITIAL_LENGTH - 2):
    #             while True:
    #                 gene = [np.random.randint(SCREEN_WIDTH), np.random.randint(SCREEN_HEIGHT)]
    #                 if gene != START_POSITION and gene != END_POSITION:
    #                     break
    #             chromo.append(gene)
    #         chromo.append(END_POSITION)
    #         self.chromosomes.append(chromo)
    #         valid_chromosomes += 1
    #     # print({valid_chromosomes}) 

    def save_epoch_results(self):
        min_fitness = min(self.fitness_scores)
        print(min_fitness)
        self.saved_data.append(min_fitness)

    def track_best_chromosome(self):
        i = np.argmin(self.fitness_scores)
        current_best_fitness = self.fitness_scores[i]
        if current_best_fitness < self.best_finess:
            self.best_finess = current_best_fitness
            self.best_chromosome = self.chromosomes[i]

        best_chromosome_index = np.argmin(self.fitness_scores)
        best_chromosome = self.chromosomes[best_chromosome_index]

    def save_best_chromosome(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{len(self.best_chromosome)}\n")
            for gene in self.best_chromosome:
                f.write(f"{gene[0]} {gene[1]}\n")

    def load_best_chromosomes(self, filename):
        with open(filename, 'r') as f:
            num_control_points = int(f.readline())

            for i in range(num_control_points):
                x_str, y_str = f.readline().split(" ")
                x, y = int(x_str), int(y_str)
                self.best_chromosome.append([x, y])
