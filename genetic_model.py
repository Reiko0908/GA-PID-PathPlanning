import matplotlib.pyplot as plt
import numpy as np
import pygame
import math

from macros import *
from bezier import *

PATH_DANGER_PRIORITIZE_FACTOR = 0.6
PATH_LENGTH_PRIORITIZE_FACTOR = 0.4
PATH_SMOOTHNESS_PRIORITIZE_FACTOR = 0.1

CHROMOSOME_CROSSOVER_SELECTION_RATIO = 0.8
ELITISM_RATIO = 0.04
POPULATION = 200
CHROMOSOME_INITIAL_LENGTH = CHROMOSOME_MAX_LENGTH // 2

def chromosome_to_bezier(chromosome):
    return

class Chromosome:
    def __init__(self):
        self.genes = []
        return

class Genetic_model:
    def __init__(self):
        print("Initializing Model")
        self.chromosomes = []

    def generate_initial_population(self):
        print("Generating Intial Chromosomes")
        for _ in range(POPULATION):
            chromosome = Chromosome().genes.append(START_POSITION) 
            while(True):


        # for _ in range(POPULATION):
        #     chromosome = np.zeros(SCREEN_HEIGHT * SCREEN_WIDTH).tolist()
        #     chromosome[CHROMOSOME_START_GENE] = 1
        #     chromosome[CHROMOSOME_END_GENE] = 1
        #
        #     num_control_points = BEZIER_ORDER + 1
        #     while(True): # Loop forever until the random genes dont match start and end genes
        #         control_genes = np.random.randint(SCREEN_HEIGHT * SCREEN_WIDTH, size = num_control_points - 2).tolist()
        #         if CHROMOSOME_START_GENE not in control_genes and CHROMOSOME_END_GENE not in control_genes:
        #             break
        #
        #     for gene in control_genes:
        #         chromosome[gene] = 1
        #
        #     self.chromosomes.append(chromosome)

    def fitness_function(self, chromosome, map):
        bezier = chromosome_to_bezier(chromosome)
        path_length = measure_bezier_length(bezier)
        path_danger = measure_bezier_danger(chromosome, map)
        fitness = (
            PATH_LENGTH_PRIORITIZE_FACTOR * path_length +
            PATH_DANGER_PRIORITIZE_FACTOR * path_danger
        )

        return fitness

    def evaluate_population(self, map):
        print("Evaluating Population")
        fitness_scores = [self.fitness_function(chromosome, map) for chromosome in self.chromosomes]
        return fitness_scores

    def select_elites(self, fitness_scores):
        num_elites = int(len(self.chromosomes) * ELITISM_RATIO)
        sorted_indices = np.argsort(fitness_scores)  
        elite_indices = sorted_indices[:num_elites]
        elites = [self.chromosomes[i] for i in elite_indices]
        return elite_indices, elites

    def crossover(self, map):
        return

    def mutation(self):
        return

    def validate(self):
        return
