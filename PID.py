import pygame
import numpy as np
import sys

from bezier import *
from macros import *
from car import *

KP = 0.5
KD = 0.1
KI = 0.01

integral = 0 
previous_Y_err = 0 

def PID(car, bezier, Kp, Ki, Kd):
    global integral, previous_Y_err

    # Tìm sai số giữa expected với real result
    projection_point = bezier.get_projection_of(car.position)
    error_vector = projection_point[0] - car.position
    Y_err = np.linalg.norm(error_vector)
    direction = np.cross(car.heading, error_vector)
    
    if direction <= 0:
        Y_err = -Y_err

    integral += Y_err / SCREEN_FPS
    derivative = (Y_err - previous_Y_err) * SCREEN_FPS

    # Tính góc lái
    steering_angle = Kp * Y_err + Ki * integral + Kd * derivative
    previous_Y_err = Y_err

    # Cập nhật góc lái và vị trí xe
    MAX_STEERING_ANGLE = np.pi / 2
    steering_angle = max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)
    car.angle += steering_angle
    car.angle_to_heading()
    car.position = car.position + CAR_VELOCITY * car.heading / SCREEN_FPS


