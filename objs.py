import numpy as np
import pygame
import math

from macros import *

MAP_END_POINTS_COLOR = (255, 0, 0)
START_POSITION = [10, 10]
END_POSITION = [SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10]

CAR_VELOCITY = 200
CAR_ANGULAR_VELOCITY = 2*np.pi/150

BEZIER_ORDER = 3
BEZIER_RESOLUTION = 50
BEZIER_LOCAL_POINT_COLOR = (255, 255, 0)
BEZIER_CONTROL_POINT_COLOR = (0, 0, 255)

NUM_OBSTACLES = 20
MAX_OBSTACLE_SIZE = 100
MIN_OBSTACLE_SIZE = 20
OBSTACLE_COLOR = (150, 150, 150)

def rotate_vect(vector, angle_radian):
    rotation_matrix = np.array([
            [np.cos(angle_radian), -np.sin(angle_radian)],
            [np.sin(angle_radian), np.cos(angle_radian)]
            ])
    return rotation_matrix @ vector

class Car:
    def __init__(self, image_path):
        image = pygame.image.load(image_path)
        self.img = pygame.transform.scale_by(image, 0.15)
        self.heading = np.array([0, -1])
        self.position = np.array(START_POSITION)

    def update(self, key):
        if key[pygame.K_d]:
            self.heading = rotate_vect(self.heading, CAR_ANGULAR_VELOCITY)
        if key[pygame.K_a]:
            self.heading = rotate_vect(self.heading, -CAR_ANGULAR_VELOCITY)
        if key[pygame.K_w]:
            self.position = self.position + CAR_VELOCITY * self.heading / SCREEN_FPS
        if key[pygame.K_s]:
            self.position = self.position - CAR_VELOCITY * self.heading / SCREEN_FPS
        return

    def draw(self, screen):
        self.heading = self.heading /np.linalg.norm(self.heading)
        north = np.array([0, -1])
        angle = np.rad2deg(np.arccos(north @ self.heading))
        if (np.cross(north, self.heading) > 0):
            image  = pygame.transform.rotate(self.img, -angle)
        else:
            image  = pygame.transform.rotate(self.img, angle)
        rect = image.get_rect(center=self.position.tolist())
        screen.blit(image, rect)
        return

# -------------------------------------------------------------------------------------------------------

class Bezier:
    def __init__(self):
        self.control_points = np.array([])
    
    def randomize(self):
        self.control_points = np.random.rand(BEZIER_ORDER + 1, 2)
        self.control_points[:, 0] = self.control_points[:, 0] * SCREEN_WIDTH
        self.control_points[:, 1] = self.control_points[:, 1] * SCREEN_HEIGHT
        return

    def update(self):
        return

    def chromosome_to_bezier(self, chromosome):
        sampled_indices = np.argwhere(chromosome == 1)
        if len(sampled_indices) != BEZIER_ORDER + 1:
            raise ValueError(f"Chromosome must encode exactly {BEZIER_ORDER + 1} control points.")

        self.control_points = np.array([
            [index % SCREEN_WIDTH, index // SCREEN_WIDTH]
            for index in sampled_indices.flatten()
        ])

    def draw(self, screen):
        for point in self.control_points:
            pygame.draw.circle(screen, BEZIER_CONTROL_POINT_COLOR, point.tolist(), 10)

        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        for t in t_values:
            local_point = np.array([0, 0])
            for i in range(BEZIER_ORDER + 1):
                local_point = local_point + math.comb(BEZIER_ORDER, i) * t**i * (1-t)**(BEZIER_ORDER-i) * self.control_points[i]
            pygame.draw.circle(screen, BEZIER_LOCAL_POINT_COLOR, local_point.tolist(), 5)
    def bezier_first_derivative(self, t):
        n = len(self.control_points) - 1  # Degree of the Bézier curve
        derivative = np.zeros(2)
        for i in range(n):
            coefficient = n * math.comb(n - 1, i)
            term = coefficient * (self.control_points[i + 1] - self.control_points[i])
            term *= t**i * (1 - t)**(n - 1 - i)
            derivative += term
            return derivative
# -------------------------------------------------------------------------------------------------------

class Obstacle:
    def __init__(self, position=None, radius=None):
        self.position = np.array(position) if position is not None else np.array([0, 0])
        self.radius = radius if radius is not None else 0

    def randomize(self):
        self.position = np.random.randint([0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.radius = np.random.randint(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)

    def draw(self, screen):
        pygame.draw.circle(screen, OBSTACLE_COLOR, self.position.tolist(), self.radius)

# -------------------------------------------------------------------------------------------------------
PATH_SAFETY_PRIORITIZE_FACTOR = 0.6
PATH_LENGTH_PRIORITIZE_FACTOR = 0.3
PATH_SMOOTHNESS_PRIORITIZE_FACTOR = 0.1

CHROMOSOME_CROSSOVER_SELECTION_RATIO = 0.8
ELITISM_RATIO = 0.04
POPULATION = 50

class Genetic_model:
    def __init__(self,map_obj):
        print("Initializing Model")
        self.chromosomes = []
        self.start_gene = START_POSITION[0] * SCREEN_HEIGHT + START_POSITION[1] # index in chormosome that indicates car start position
        self.end_gene = END_POSITION[0] * SCREEN_HEIGHT + END_POSITION[1] # index in chormosome that indicates car end position
        self.chromosome_length = SCREEN_HEIGHT * SCREEN_WIDTH # length of each chromosome
        self.map = map_obj  # Map object containing the danger map

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
    def chromosome_to_bezier(self, chromosome):
        bezier = Bezier()
        bezier.chromosome_to_bezier(chromosome) 
        return bezier

    def measure_bezier_length(self, bezier_curve):
        if len(bezier_curve.control_points) < 2:
            return 0.0

        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        total_length = 0.0

        for i in range(len(t_values) - 1):
            t1, t2 = t_values[i], t_values[i + 1]
            p1 = bezier_curve.bezier_first_derivative(t1)
            p2 = bezier_curve.bezier_first_derivative(t2)

            # Approximate integral using the trapezoidal rule
            avg_magnitude = (np.linalg.norm(p1) + np.linalg.norm(p2)) / 2
            total_length += avg_magnitude * (t2 - t1)
        return total_length    

    def measure_bezier_danger(self, chromosome):
        danger_map = self.map.danger_map
        active_genes = [i for i, gene in enumerate(chromosome) if gene == 1]
        if len(active_genes) <= 2:
            return 0
        active_genes = active_genes[1:-1]
        total_danger = sum(
                danger_map[i // SCREEN_WIDTH, i % SCREEN_WIDTH] for i in active_genes
        )
        average_danger = total_danger / len(active_genes)
        return average_danger


    def calculate_bezier_smoothness(self, bezier):
        bezier_points = bezier.sample_points()
        smoothness = 0
        for i in range(1, len(bezier_points) - 1):
            p1, p2, p3 = bezier_points[i - 1], bezier_points[i], bezier_points[i + 1]
            vec1 = p2 - p1
            vec2 = p3 - p2
            angle = np.arccos(
                np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1, 1)
            )
            smoothness += angle
        return smoothness

    def fitness_function(self, chromosome):
        bezier = self.chromosome_to_bezier(chromosome)
        # Measure path length
        path_length = self.measure_bezier_length(chromosome)

        # Measure danger
        path_danger = self.measure_bezier_danger(chromosome)
        # Measure smoothness
        path_smoothness = self.calculate_bezier_smoothness(bezier)
        # Combine metrics into the fitness score
        length_factor = PATH_LENGTH_PRIORITIZE_FACTOR
        danger_factor = PATH_SAFETY_PRIORITIZE_FACTOR
        smoothness_factor = PATH_SMOOTHNESS_PRIORITIZE_FACTOR  # Adjust weight for smoothness as needed

        fitness = (
            length_factor * path_length +
            danger_factor * path_danger +
            smoothness_factor * path_smoothness
        )
        return fitness

    def evaluate_population(self):
        """
        Evaluate the fitness of the entire population.
        """
        fitness_scores = [self.fitness_function(chromosome) for chromosome in self.chromosomes]
        return fitness_scores

    def select_elites(self, fitness_scores):
        num_elites = int(len(self.chromosomes) * ELITISM_RATIO)
        sorted_indices = np.argsort(fitness_scores)  
        elite_indices = sorted_indices[:num_elites]
        elites = [self.chromosomes[i] for i in elite_indices]
        return elite_indices, elites

    def crossover(self):
        print("Performing Crossover")
        fitness_scores = self.evaluate_population()
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
