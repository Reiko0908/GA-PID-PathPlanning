import pygame as pg, math, time
import numpy as np
import random as rd

# General variables:
WIDTH = 1550
HEIGHT = 800
running = True
G = 0.3

# environment initialization
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Circle-rotation game")
clock = pg.time.Clock()
font = pg.font.SysFont('Comic Sans MS', 20)

# Number of balls in each color:
def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

red, blue, green, purple, brown, white, yellow, pink, orange = 0, 0, 0, 0, 0, 0, 0, 0, 0
color_name = ["red", "blue", "green", "purple", "brown", "white", "yellow", "pink", "orange"]

class Rotating_Circle():
    def __init__(self):
        self.RADIUS = 300
        self.center = np.array([WIDTH/2, HEIGHT/2], dtype=np.float64)
        self.start_angle = 30
        self.end_angle = -30
        self.rotating_speed = -2 
    def draw_circle(self):
        self.body = pg.draw.circle(screen, "yellow", self.center, self.RADIUS, 5)
        self.first_side = self.center + (self.RADIUS + 200)*np.array([math.cos(math.radians(self.start_angle)),
                                                  math.sin(math.radians(self.start_angle))], dtype=np.float64)
        self.second_side = self.center + (self.RADIUS + 200)*np.array([math.cos(math.radians(self.end_angle)), 
                                                   math.sin(math.radians(self.end_angle))], dtype=np.float64)
        pg.draw.polygon(screen, "black", [self.center, self.first_side, self.second_side])
        self.start_angle += self.rotating_speed
        self.end_angle +=  self.rotating_speed
        self.start_angle %= 360
        self.end_angle %= 360

class Ball():
    def __init__(self):
        self.pos = np.array([WIDTH/2 + rd.randint(-200, 200), HEIGHT/2 - 190], dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)
        self.color = rd.choice(color_name)
        self.SIZE = 10
        self.is_in = True
    
    def draw_ball(self):
        self.ball = pg.draw.circle(screen, self.color, self.pos, self.SIZE)
        self.vel[1] += G
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
    
    def collision_check(self, circle_center : list, circle_radius):
        self.pos_vector = self.pos - circle_center
        self.distance = np.linalg.norm(self.pos_vector)
        self.tangential_vector = np.array([-self.pos_vector[1], self.pos_vector[0]], dtype=np.float64)
        if abs(self.distance - circle_radius) < 10:
            return True
        else:
            return False

    def keep_inside(self, circle_center, circle_radius):
        self.pos = circle_center + (circle_radius - self.SIZE)*(self.pos_vector/self.distance)

    def bounce(self):
        bounce_vector = (np.dot(self.vel, self.tangential_vector)/np.dot(self.tangential_vector, self.tangential_vector))*self.tangential_vector
        self.vel = 2*bounce_vector - self.vel
        self.vel += 0.2
    
    def outside_check(self, triangle: np.array):
        check = 0
        AB, AP = triangle[1] - triangle[0], self.pos - triangle[0] - 5
        cross_1 = cross_product(AB, AP)
        BC, BP = triangle[2] - triangle[1], self.pos - triangle[1] - 5
        cross_2 = np.cross(BC, BP)
        CA, CP = triangle[0] - triangle[2], self.pos - triangle[2] - 5
        cross_3 = np.cross(CA, CP)
        return (cross_1 >= check and cross_2 >= check and cross_3 >= check or 
                cross_1 <= check and cross_2 <= check and cross_3 <= check)

def balls_record():
    text_surf = font.render("Number of balls: ", True, "yellow")
    screen.blit(text_surf, text_surf.get_rect(center = (90,20)))
    text_surf = font.render(str(number_of_balls), True, "red")
    screen.blit(text_surf, text_surf.get_rect(center =(170 + len(str(number_of_balls)*4),20)))

def color_record():
    index = 1
    for i in ball_color:
        text_surf = font.render(f"{color_name[index - 1]} :", True, "yellow")
        screen.blit(text_surf, text_surf.get_rect(center = (90,20 + 30*index)))
        text_surf = font.render(str(i), True, "red")
        screen.blit(text_surf, text_surf.get_rect(center = (130 + 5*len(str(i)),20 + 30*index)))
        index += 1
    for j in ball_color:
        j = 0

if __name__ == "__main__":
    circle = Rotating_Circle()
    first_ball = Ball()
    list_of_ball = [first_ball]
    while running:
        ball_color = [red, blue, green, purple, brown, white, yellow, pink, orange]
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print(ball_color)
                running = False
            if event.type == pg.K_1:
                time.sleep(10)
                running == False
        screen.fill("black")
        circle.draw_circle()
        for ball in list_of_ball:
            if ball.is_in == True:
                ball_color[color_name.index(ball.color)] += 1
            ball.draw_ball()
            if Ball.collision_check(ball, circle.center, circle.RADIUS):
                if Ball.outside_check(ball, [circle.center, circle.first_side, circle.second_side]):
                    ball.is_in = False
                else:
                    ball.keep_inside(circle.center, circle.RADIUS)
                    ball.bounce()
            if ball.pos[0] > WIDTH or ball.pos[0] < 0 or ball.pos[1] > HEIGHT or ball.pos[1] < 0:
                list_of_ball.remove(ball)
                list_of_ball.append(Ball())
                list_of_ball.append(Ball())
        number_of_balls = len(list_of_ball)
        balls_record()
        color_record()
        pg.display.flip()
        clock.tick(24)
    pg.quit()
    print(ball_color)
