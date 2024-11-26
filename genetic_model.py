import matplotlib.pyplot as plt
import numpy as np
import pygame
import math

from macros import *
from bezier import *

PATH_DANGER_PRIORITIZE_FACTOR = 0.6
PATH_LENGTH_PRIORITIZE_FACTOR = 0.4
PATH_SMOOTHNESS_PRIORITIZE_FACTOR = 0.1

CROSSOVER_RATIO = 0.5
ELITISM_RATIO = 0.3
MUTATION_RATIO = 0.1
POPULATION = 100
CHROMOSOME_INITIAL_LENGTH = 10

def chromosome_to_bezier(chromosome):
    bezier = Bezier()
    bezier.control_points= np.array([
            gene
            for gene in chromosome
            ])
    return bezier

def fitness_function(chromosome, map):
    bezier = chromosome_to_bezier(chromosome)
    path_length = bezier.get_length()
    path_danger = measure_bezier_danger(chromosome, map)
    fitness = (
        PATH_LENGTH_PRIORITIZE_FACTOR * path_length +
        PATH_DANGER_PRIORITIZE_FACTOR * path_danger
    )
    return fitness

class Genetic_model:
    def __init__(self):
        print("Initializing Model")
        self.chromosomes = []
        self.fitness_scores = []
        self.elite_indices = []

    def generate_initial_population(self):
        print("Generating Intial Chromosomes")
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
        print("Evaluating Population")
        self.fitness_scores = [fitness_function(chromosome, map) for chromosome in self.chromosomes]

    def select_elites(self):
        num_elites = int(len(self.chromosomes) * ELITISM_RATIO)
        sorted_indices = np.argsort(self.fitness_scores)  
        self.elite_indices = sorted_indices[:num_elites]
        # elites = [self.chromosomes[i] for i in elite_indices]

    def crossover(self):
        print("Performing Crossover")

        non_elite_indices = np.setdiff1d(np.arange(len(self.chromosomes)), self.elite_indices)
        num_crossover = int(len(non_elite_indices) * CROSSOVER_RATIO)
        chosen_parent_indices = np.random.choice(
                len(non_elite_indices), num_crossover, replace=False
                )
        for i in range(len(chosen_parent_indices - 1), 2):
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
        position = np.random.randint(0, len(self.chromosomes[chromo_index]) - 1) 
        new_gene = [np.random.randint(SCREEN_WIDTH), np.random.randint(SCREEN_HEIGHT)]
        self.chromosomes[chromo_index].insert(position, new_gene)

    def mutate_remove_gene(self, chromo_index):
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        del self.chromosomes[chromo_index][position]

    def mutate_edit_gene(self, chromo_index):
        position = np.random.randint(1, len(self.chromosomes[chromo_index]) - 1) 
        self.chromosomes[chromo_index][position] = [
                np.random.randint(SCREEN_WIDTH),
                np.random.randint(SCREEN_HEIGHT)
                ]

    def mutate(self):
        print("Performing Mutation")

        mutate_chosen = np.random.choice([True, False], size = POPULATION, p=[MUTATION_RATIO, 1-MUTATION_RATIO])
        # print([i for i, x in enumerate(mutate_chosen) if x == True])
        for chromo_index in range(POPULATION):
            if not mutate_chosen[chromo_index]:
                continue
            mutate_type = np.random.randint(3)
            if mutate_type == 0:
                self.mutate_edit_gene(chromo_index)
            elif mutate_type == 1:
                self.mutate_add_gene(chromo_index)
            else:
                self.mutate_remove_gene(chromo_index)


    def validate(self):

       return

