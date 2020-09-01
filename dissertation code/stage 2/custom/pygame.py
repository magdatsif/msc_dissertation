import pygame, sys 
import os
from pygame.locals import *
import math
import random
# Implements pygame for headless systems 
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# pygame.display.init()


window_width = 640
window_height = 360
window_size = (window_width, window_height)
   
# Determines the position of the object in the 5x1 grid
def quantize_y_axis(object):
    bin_y = 0
    if 0 <= object.posy + object.image_height/2 <= window_height/5:
        bin_y = 0 
    elif window_height/5 < object.posy + object.image_height/2 <= 2*window_height/5:
        bin_y = 1
    elif 2*window_height/5 < object.posy + object.image_height/2 <= 3*window_height/5:
        bin_y = 2
    elif 3*window_height/5 < object.posy + object.image_height/2 <= 4*window_height/5:
        bin_y = 3 
    elif 4*window_height/5 < object.posy + object.image_height/2 <= window_height:
        bin_y = 4 

    return bin_y

# Agent's class
class Satellite:
    def __init__(self, posx, posy):
        self.posx = posx
        self.posy = posy
        
        self.image = pygame.image.load('satellite4.png')
        self.image_width, self.image_height = self.image.get_rect().size

        self.image = pygame.transform.scale(self.image, (int(self.image_width/1.2), int(self.image_height/1.2)))
        self.image_width, self.image_height = self.image.get_rect().size

        self.collision = False
        self.is_alive = True
        self.took_action = False
        
        self.reward = 0
        self.reward_given = False

        self.objects_met = 0
        self.objects_avoided = 0
        self.object_target = 20

        self.collision_posy = 0

    # Action space of the satellite
    def move_up(self):
        if not self.posy < 10:
            self.posy -= 10

    def move_down(self):
        if not (window_height - self.image_height) < (self.posy-10):
            self.posy += 10

    def satellite_safe(self, other):
        if other.posx < window_width:
            return not satellite_aligned_w_obstacle(self, other)

    def action(self, action):
        if action == 0:
            self.move_up()
        if action == 1:
            self.move_down()
        if action == 2:
            self.posy = self.posy
    
    # Renders the satellite
    def draw(self, window):
        window.blit(self.image, [self.posx, self.posy])

    # Returns satellite's position in the 5x1 grid
    def position_in_quantised_space(self):
        return quantize_y_axis(self)
    
    # Checks if a satellite avoided the collision with an obstacle
    def not_collision(self, other):
        if self.posx < other.posx < self.posx + self.image_width or self.posx < other.posx + other.image_width < self.posx + self.image_width:
            if not self.satellite_aligned_w_obstacle(other):
                return True
    
    # Checks if the satellite's trajectory is aligned with the obstacle's
    def satellite_aligned_w_obstacle(self, obstacle):
        if obstacle.posy < self.posy < obstacle.posy + obstacle.image_height:
            return True
        if self.posy < obstacle.posy < self.posy + self.image_height:
            return True

# Obstacle's class
class Debris:
    def __init__(self, posx):
        self.image = pygame.image.load('debris1.png')
        self.posx = posx
        self.image_width, self.image_height = self.image.get_rect().size
        self.image = pygame.transform.scale(self.image, (int(self.image_width/1.45), int(self.image_height/1.45)))
        self.image_width, self.image_height = self.image.get_rect().size
        
        # Position of obstacle either up or down in the window
        self.rand = random.randint(0,1)
        if self.rand:
            self.posy = 0 
        else:
            self.posy = window_height- self.image_height

        self.checked = False
        self.out_of_window = False
        self.reward_given = False
        self.reward = 0

    # Renders debris
    def draw(self, window):
        if(self.posx > -2):
            self.posx -= 4
            window.blit(self.image, (self.posx, self.posy))
        else:
            self.out_of_window = True
    
    # Moves debris
    def update(self):
        if(self.posx > -2):
            self.posx -= 4
        else:
            self.out_of_window = True
    
    # Returns debris's position in the 5x1 grid
    def position_in_quantised_space(self):
        return quantize_y_axis(self)
   
    # Checks if the obstacle object collided with the satellite
    def had_collision(self, satellite):
        if satellite.posx < self.posx < (satellite.posx + satellite.image_width):
            if satellite.satellite_aligned_w_obstacle(self):
                satellite.collision_posy = self.posy
                return True
   
    # Action space of the obstacle
    def action(self, action):
        if action == 0:
            self.move_up()
        if action == 1:
            self.move_down()
        if action == 2:
            self.posy = self.posy

    def move_up(self):
        if not self.posy < 10:
            self.posy -= 2

    def move_down(self):
        if not (window_height - self.image_height) < (self.posy-10):
            self.posy += 2

# Background's class
class Background():
    def __init__(self):
        self.image = pygame.image.load('sky1.jpeg')
        self.image_rect = self.image.get_rect()
        self.image = pygame.transform.scale(self.image, (int(window_size[0] / self.image_rect.width * self.image_rect.width), int(window_size[0] / self.image_rect.width * self.image_rect.height)))
        self.image_rect = self.image.get_rect()

        self.posx1 = 0
        self.posy1 = 0
        self.posx2 = self.image_rect.width
        self.posy2 = 0

        self.velocity_x = 3
    
    # Method for scrolling the background 
    def update(self):
        self.posx1 -= self.velocity_x
        self.posx2 -= self.velocity_x
        if self.posx1 <= -self.image_rect.width:
            self.posx1 = self.image_rect.width
        if self.posx2 <= -self.image_rect.width:
            self.posx2 = self.image_rect.width
    
    # Renders the background
    def draw(self, window):
        window.blit(self.image, (self.posx1, self.posy1))
        window.blit(self.image, (self.posx2, self.posy2))

# Pygame class
class PyGameObjectAvoidance:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('DebrisAvoidance-v1')
        pygame.time.set_timer(USEREVENT, random.randrange(2000, 2001))
        pygame.font.get_fonts()
        

        self.explosion_img = pygame.image.load('collision.png')
        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.satellite = Satellite(10, 160)
        self.game_speed = 1000
        self.mode = 0
        self.background = Background()
        self.debris = []
        self.debris_rewards = []
        self.font = pygame.font.SysFont('freesansbold.ttf', 32)

    # shows progress of agent
    def scoreboard(self, window):
        self.progress = self.font.render("Obstacles avoided: " + str(self.satellite.objects_avoided) + "/15", True, (255, 255, 255))
        self.progress_rect = self.progress.get_rect()
        self.progress_rect.center = (window_width - 0.6*self.progress_rect.size[0], window_height - self.progress_rect.size[1])
        self.window.blit(self.progress, self.progress_rect)
    
    # Renders the environment
    def view(self):

        self.background.update()
        self.background.draw(self.window)

        if self.satellite.is_alive:
            self.satellite.draw(self.window)
            
            for d in self.debris:
                d.draw(self.window)

        else:
            self.window.blit(self.explosion_img, (self.satellite.posx, self.satellite.collision_posy))

        self.scoreboard(self.window)

        events = pygame.event.get()

        pygame.display.flip()
        self.clock.tick(self.game_speed)
