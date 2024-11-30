import matplotlib.pyplot as plt
import numpy as np
import pygame
import math
import sys

from macros import *

BEZIER_RESOLUTION = 50
BEZIER_LOCAL_POINT_COLOR = (0, 0, 255)
BEZIER_CONTROL_POINT_COLOR = (0, 0, 255)
BEZIER_SPLINE_COLOR = (72, 61, 139)
BEZIER_SPLINE_WIDTH = 5

BINARY_SEARCH_STOP_THRESHOLD = 1E-5

class Bezier:
    def __init__(self):
        self.control_points = np.array([])
    
    def randomize(self, order):
        self.control_points = np.random.rand(order + 1, 2)
        self.control_points[:, 0] = self.control_points[:, 0] * SCREEN_WIDTH
        self.control_points[:, 1] = self.control_points[:, 1] * SCREEN_HEIGHT
        return

    def update(self):
        return

    def calculate_local_point(self, t):
        num_control_points = len(self.control_points)
        local_point = np.array([0, 0])
        for i in range(num_control_points):
            bernstein_coeff = math.comb(num_control_points-1, i) * (1 - t)**(num_control_points-i-1) * t**i
            local_point = local_point + bernstein_coeff * self.control_points[i]
        return local_point


    def game_draw(self, screen):
        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        local_points = [self.calculate_local_point(t) for t in t_values]
        for i in range(1, BEZIER_RESOLUTION):
            pygame.draw.line(screen, BEZIER_SPLINE_COLOR, local_points[i-1], local_points[i], BEZIER_SPLINE_WIDTH)

    def calculate_first_derivative(self, t):
        derivative = np.zeros(2)
        bezier_order = len(self.control_points) - 1
        for i in range(bezier_order):
            coefficient = bezier_order * math.comb(bezier_order - 1, i)
            term = coefficient * (self.control_points[i + 1] - self.control_points[i])
            term = term * t**i * (1 - t)**(bezier_order - 1 - i)
            derivative += term
        return derivative

    def get_projection_of(self, target_point):
        t, t_left, t_right = 0, 0, 0
        flag = 0
        min_dist = sys.float_info.max

        t_values = np.linspace(0, 1, BEZIER_RESOLUTION)
        for i in range(BEZIER_RESOLUTION):
            local_point = self.calculate_local_point(t_values[i])
            dist = np.linalg.norm(local_point - target_point)
            if dist < min_dist:
                min_dist = dist
                flag = i

        if flag == BEZIER_RESOLUTION - 1:
            t_left = t_values[-2]
            t = 1.0
            t_right = 1.0
        elif flag == 0:
            t_right = t_values[1]
            t = 0.0
            t_left = 0.0
        else:
            t = t_values[flag]
            t_left = t_values[flag - 1]
            t_right = t_values[flag + 1]

        projection_length = 0
        while(t_right - t_left > BINARY_SEARCH_STOP_THRESHOLD):
            t1, t2 = (t + t_left) / 2, (t + t_right) / 2
            p1, p2 = self.calculate_local_point(t1), self.calculate_local_point(t2)
            norm1, norm2 = np.linalg.norm((p1 - target_point)), np.linalg.norm((p2 - target_point))
            projection_length = min(norm1, norm2)
            if(norm1 < norm2):
                t_right = t
                t = t1
            else:
                t_left = t
                t = t2

        return self.calculate_local_point(t), projection_length


    def get_length(self,bezier):
        t_values = np.linspace(0, 1,BEZIER_RESOLUTION)
        sampled_points = np.array([bezier.calculate_local_point(t) for t in t_values])
        # Calculate the distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(sampled_points, axis=0) ** 2, axis=1))
        # Sum up the distances to approximate the curve length
        total_length = np.sum(distances)
        return total_length
