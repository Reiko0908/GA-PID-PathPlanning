import matplotlib.pyplot as plt
import numpy as np
import pygame
import math

from macros import *

BEZIER_ORDER = 3
BEZIER_RESOLUTION = 50
BEZIER_LOCAL_POINT_COLOR = (0, 0, 255)
BEZIER_CONTROL_POINT_COLOR = (0, 0, 255)

def measure_bezier_length(bezier_curve):
    if len(bezier_curve.control_points) < 2:
        return 0.0

    t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
    total_length = 0.0

    for i in range(BEZIER_RESOLUTION - 1):
        t1, t2 = t_values[i], t_values[i + 1]
        p1 = bezier_curve.first_derivative(t1)
        p2 = bezier_curve.first_derivative(t2)

        # Approximate integral using the trapezoidal rule
        avg_magnitude = (np.linalg.norm(p1) + np.linalg.norm(p2)) / 2
        total_length += avg_magnitude * (t2 - t1)
    return total_length    

def calculate_bezier_smoothness(bezier):
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

def measure_bezier_danger(chromosome, map):
    danger_map = map.danger_map
    active_genes = [i for i, gene in enumerate(chromosome) if gene == 1]
    if len(active_genes) <= 2:
        return 0
    active_genes = active_genes[1:-1]
    total_danger = sum([
            danger_map[i // SCREEN_WIDTH, i % SCREEN_WIDTH] for i in active_genes
            ])
    average_danger = total_danger / len(active_genes)
    return average_danger

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

    def game_draw(self, screen):
        for point in self.control_points:
            pygame.draw.circle(screen, BEZIER_CONTROL_POINT_COLOR, point.tolist(), 10)

        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        for t in t_values:
            local_point = np.array([0, 0])
            for i in range(BEZIER_ORDER + 1):
                local_point = local_point + math.comb(BEZIER_ORDER, i) * t**i * (1-t)**(BEZIER_ORDER-i) * self.control_points[i]
            pygame.draw.circle(screen, BEZIER_LOCAL_POINT_COLOR, local_point.tolist(), 5)

    def first_derivative(self, t):
        derivative = np.zeros(2)
        for i in range(BEZIER_ORDER):
            coefficient = BEZIER_ORDER * math.comb(BEZIER_ORDER - 1, i)
            term = coefficient * (self.control_points[i + 1] - self.control_points[i])
            term = term * t**i * (1 - t)**(BEZIER_ORDER - 1 - i)
            derivative += term
            return derivative

