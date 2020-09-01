import math
import numpy as np
import gym
from gym import spaces
from gym.envs.custom.pygame import PyGameObjectAvoidance
from gym.envs.custom.pygame import Debris
from gym.utils import seeding
import pygame, sys 
from pygame.locals import *
import random


window_width = 640
window_height = 360

# Custom Gym Environment
class SatelliteEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.game = PyGameObjectAvoidance()
        self.state = None

        # Boundaries of observation space
        self.low = np.array([0, 0, 0, 0, 0], dtype=np.int)
        self.high = np.array([4, 4, 4, 4, 4], dtype=np.int)

        # Defining the observation and action spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int)

        # self.flag_move_deb = None
        self.newpos = None
    
    # Resets the environment
    def reset(self):
        del self.game
        self.game = PyGameObjectAvoidance()

        # Makes a list of obstacles, 200-300pixels away from one another 
        self.newpos = 0
        # self.flag_move_deb = 0
        for i in range(50):
            rand = random.randint(0,2)
            self.game.debris.append(Debris(window_width + self.newpos))
            self.newpos += np.random.randint(200,350)

        # Updates the obstacles so they can move
        for i in self.game.debris:
            i.update()

        # Returns the initial state of the environment, 3-tuple of the positions of the satellite and the 2 closest obstacles
        self.state = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space()[0], self.game.debris[0].position_in_quantised_space()[1] , self.game.debris[1].position_in_quantised_space()[0], self.game.debris[1].position_in_quantised_space()[1])
        return np.array(self.state)

    # One-step function of the environment
    def step(self, action):
        # Implements movement of the debris as the DQN network was having issues
        for j in self.game.debris:
            if(j.posx > -2):
                j.posx -= 4
            else:
                j.out_of_window = True
                self.game.debris.append(Debris(window_width + self.newpos))
                self.newpos += np.random.randint(200,350)

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # The agent observes current state
        obs = self.state
        reward = 0
        done = False

        # The agent takes the action
        self.game.satellite.action(action)


        # Observe state at t+1
        obs = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space()[0], self.game.debris[0].position_in_quantised_space()[1] , self.game.debris[1].position_in_quantised_space()[0], self.game.debris[1].position_in_quantised_space()[1])
    

        # For the nearest obstacle, check if it is out of the environment
        i = self.game.debris[0]
        
        if i.out_of_window:
            self.game.debris.pop(self.game.debris.index(i))

        # Determine the reward
        if i.had_collision(self.game.satellite):
            reward = -100
            self.game.satellite.is_alive = False
        elif self.game.satellite.not_collision(i):
            reward = 10
        elif self.game.satellite.satellite_aligned_w_obstacle(i):
            reward = -1 

        # Updates scoreboard
        if self.game.satellite.is_alive and i.out_of_window:
            if not i.checked:                            
                self.game.satellite.objects_avoided += 1
                i.checked = True
                
        # If satellite is dead the environment needs to reset
        if not self.game.satellite.is_alive:
            done = True

        return obs, reward, done, {}
    # Render the environment

    def render(self, mode="human", close=False):
        self.game.view()