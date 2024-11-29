import matplotlib.pyplot as plt
import numpy as np
import pygame
import math

from macros import *
from bezier import *

PATH_DANGER_PRIORITIZE_FACTOR = 0.8
PATH_LENGTH_PRIORITIZE_FACTOR = 0.2

CROSSOVER_RATIO = 0.5
ELITISM_RATIO = 0.1
MUTATION_RATIO = 0.2
POPULATION = 200
CHROMOSOME_INITIAL_LENGTH = 7
NUM_EPOCH = 100

def chromosome_to_bezier(chromosome):
    bezier = Bezier()
    bezier.control_points= np.array([gene for gene in chromosome])
    return bezier

def normalize_array(array):
    min_value = min(array)
    max_value = max(array)
    for i in range(len(array)):
        array[i] = (array[i] - min_value) / (max_value - min_value)

def fitness_function(normalize, danger):
    if danger >= 0.1:
        return 1
    fitness = (
        PATH_LENGTH_PRIORITIZE_FACTOR * normalize +
        PATH_DANGER_PRIORITIZE_FACTOR * danger
    )
    return fitness

class Genetic_model:
    def __init__(self):
        # print("Initializing Model")
        self.chromosomes = []
        self.fitness_scores = []
        self.elite_indices = []
        self.saved_data = []

    def generate_initial_population(self):
        # print("Generating Intial Chromosomes")
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
        # print("Evaluating Population")
        self.fitness_scores = [0] * len(self.chromosomes)
        bezier_lengths = []
        bezier_dangers = []
        for chromo in self.chromosomes:
            bezier = chromosome_to_bezier(chromo)
            length = bezier.get_length(bezier)
            danger = measure_bezier_danger(bezier, map)
            
            bezier_lengths.append(length)
            bezier_dangers.append(danger)

        normalize_array(bezier_lengths)
        for i in range(len(self.chromosomes)):
            self.fitness_scores[i] = fitness_function(bezier_lengths[i], bezier_dangers[i])

    def select_elites(self):
        num_elites = int(len(self.chromosomes) * ELITISM_RATIO)
        sorted_indices = np.argsort(self.fitness_scores)  
        self.elite_indices = sorted_indices[:num_elites]
        self.non_elite_indices = sorted_indices[num_elites:] 

    def crossover(self):
        # print("Performing Crossover") 
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
        # Access the actual chromosome using the index from non_elite_indices
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        del self.chromosomes[chromo_index][position]

    def mutate_edit_gene(self, chromo_index):
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        self.chromosomes[chromo_index][position] = [
                np.random.randint(SCREEN_WIDTH),
                np.random.randint(SCREEN_HEIGHT)
            ]

    def mutate(self):
        # print("Performing Mutation")
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

    def validate(self, map): 
        # print("Validating Population")
        valid_chromosomes = []
    
        for chromo in self.chromosomes:
            bezier = chromosome_to_bezier(chromo)
            valid = True
            for obs in map.obstacles:
                _, proj_length = bezier.get_projection_of(obs.position)
                if proj_length <= obs.radius:  # Check if chromosome is invalid
                    valid = False
                    break
        
            if valid:
                valid_chromosomes.append(chromo)  # Only keep valid chromosomes

        # Replace the old population with the validated one
        self.chromosomes = valid_chromosomes

    def full_fill_population(self):
        valid_chromosomes = len(self.chromosomes)
        while valid_chromosomes < POPULATION:
            chromo = [START_POSITION]
            for _ in range(CHROMOSOME_INITIAL_LENGTH - 2):
                while True:
                    gene = [np.random.randint(SCREEN_WIDTH), np.random.randint(SCREEN_HEIGHT)]
                    if gene != START_POSITION and gene != END_POSITION:
                        break
                chromo.append(gene)
            chromo.append(END_POSITION)
            self.chromosomes.append(chromo)
            valid_chromosomes += 1
        # print({valid_chromosomes}) 

    def save_epoch_results(self):
        min_fitness = min(self.fitness_scores)
        print(min_fitness)
        self.saved_data.append(min_fitness)

    def save_best_chromosome(self, filename, generation):
        # Find the index of the best chromosome based on fitness scores
        best_chromosome_index = np.argmin(self.fitness_scores)  # Minimize fitness score
        best_chromosome = self.chromosomes[best_chromosome_index]
    
        # Save the best chromosome to a file with the generation number
        with open(filename, 'a') as f:
            f.write(f"Generation {generation}: ")
            f.write(' '.join(map(str, [item for gene in best_chromosome for item in gene])) + '\n')

    def load_best_chromosomes(self, filename):
        best_chromosomes = []
    
        # Read the best chromosomes from the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
            # Parse each line and extract the chromosomes
            for line in lines:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(': ')
                    chromosome_data = list(map(int, parts[1].split()))
                    # Reconstruct the chromosome (assuming gene pairs)
                    chromosome = [chromosome_data[i:i + 2] for i in range(0, len(chromosome_data), 2)]
                    best_chromosomes.append(chromosome)
    
        return best_chromosomes
