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
POPULATION = 50

def chromosome_to_bezier(chromosome):
    bezier = Bezier()
    sampled_indices = [i for i, x in enumerate(chromosome) if x == 1]

    if len(sampled_indices) != BEZIER_ORDER + 1:
        raise ValueError(f"Chromosome must encode exactly {BEZIER_ORDER + 1} control points.")
    bezier.control_points = np.array([
        [index % SCREEN_WIDTH, index // SCREEN_WIDTH]
        for index in sampled_indices
    ])

    return bezier

class Genetic_model:
    def __init__(self):
        print("Initializing Model")
        self.chromosomes = []
        self.start_gene = START_POSITION[0] * SCREEN_HEIGHT + START_POSITION[1] # index in chormosome that indicates car start position
        self.end_gene = END_POSITION[0] * SCREEN_HEIGHT + END_POSITION[1] # index in chormosome that indicates car end position
        self.chromosome_length = SCREEN_HEIGHT * SCREEN_WIDTH # length of each chromosome

    def generate_initial_population(self):
        print("Generating Intial Chromosomes")
        for _ in range(POPULATION):
            chromosome = np.zeros(SCREEN_HEIGHT * SCREEN_WIDTH).tolist()
            chromosome[self.start_gene] = 1
            chromosome[self.end_gene] = 1

            num_control_points = BEZIER_ORDER + 1
            while(True): # Loop forever until the random genes dont match start and end genes
                control_genes = np.random.randint(SCREEN_HEIGHT * SCREEN_WIDTH, size = num_control_points - 2).tolist()
                if self.start_gene not in control_genes and self.end_gene not in control_genes:
                    break
            
            for gene in control_genes:
                chromosome[gene] = 1

            self.chromosomes.append(chromosome)

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
        print("Performing Crossover")
        fitness_scores = self.evaluate_population(map)
        elite_indices, elites = self.select_elites(fitness_scores)
        non_elite_indices = np.setdiff1d(np.arange(len(self.chromosomes)), elite_indices)        

        num_crossover = int(len(non_elite_indices)*CHROMOSOME_CROSSOVER_SELECTION_RATIO)
        selected_indices = np.random.choice(
                len(non_elite_indices), num_crossover, replace=False
        )
        selected_chromosomes = [self.chromosomes[i] for i in selected_indices]
        np.random.shuffle(selected_chromosomes)
        
        new_chromosomes = []
        for i in range (0, len(selected_chromosomes) -1, 2):
            parent1 = selected_chromosomes[i]
            parent2 = selected_chromosomes[i+1]

            points = np.random.choice(
                range(1, self.chromosome_length - 1),2, replace=False
            )
            point1, point2 = sorted(points)

            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            
            child1[self.start_gene] = 1
            child1[self.end_gene] = 1
            child2[self.start_gene] = 1
            child2[self.end_gene] = 1

            new_chromosomes.extend([child1, child2])

            num_replacement = len(self.chromosomes) - len(elites)
            self.chromosomes = elites + new_chromosomes[:num_replacement]

    def mutation(self):
        generation_mutate_ratio = np.random.random()
        chromosome_mutate_ratio = np.random.uniform(0.1, 0.5)
        gene_mutate_ratio = np.random.uniform(0.02,0.3)

        if generation_mutate_ratio < chromosome_mutate_ratio:
            print("Mutation skipped for this generation.")
            return
        print("Performing mutation for this generation.")
        num_chromosome_mutate = int(len(self.chromosomes) * chromosome_mutate_ratio)
        selected_chromosomes_indices = np.random.choice(
                len(self.chromosomes), num_chromosome_mutate, replace=False
        )
        for idx in selected_chromosomes_indices: 
            chromosome = self.chromosomes[idx]
            num_gene_mutate = int(self.chromosome_length * gene_mutate_ratio)
            selected_genes_indices = np.random.choice(
                    range(1, self.chromosome_length - 1), num_gene_mutate, replace=False
        )
            for gene_idx in selected_genes_indices:
                chromosome[gene_idx] = 1 - chromosome[gene_idx]

                self.chromosomes[idx] = chromosome

    def validate(self):
        return
