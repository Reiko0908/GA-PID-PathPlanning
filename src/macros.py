import numpy as np

#SCREEN
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
SCREEN_FPS = 60

# OBSTACLE
NUM_OBSTACLES = 20
MAX_OBSTACLE_SIZE = 60
MIN_OBSTACLE_SIZE = 10
OBSTACLE_COLOR = (150, 150, 150)

# MAP
MAP_END_POINTS_COLOR = (255, 0, 0)
START_POSITION = [10, 10]
END_POSITION = [SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10]

# BEZIER
BEZIER_RESOLUTION = 50
BINARY_SEARCH_STOP_THRESHOLD = 1E-5

# CAR
CAR_VELOCITY = 200
CAR_ANGULAR_VELOCITY = 2*np.pi/150
MAX_OMEGA_NOISE = 0.1
NOISE_FREQUENCY = 10

# GENETIC MODEL
CROSSOVER_RATIO = 0.5
SEPARATION_THRESHOLD = 0.25
MUTATION_RATIO = 0.2
POPULATION = 200
CHROMOSOME_INITIAL_LENGTH = 7
NUM_EPOCH = 100
