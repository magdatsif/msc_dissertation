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
        self.state_satellite = None
        self.state_debris = None

        # Boundaries of observation spaces
        self.low_satellite = np.array([0, 0, 0], dtype=np.int)
        self.high_satellite = np.array([4, 4, 4], dtype=np.int)

        self.low_debris = np.array([0, 0], dtype=np.int)
        self.high_debris = np.array([4, 4], dtype=np.int)

        # Defining the observation and action spaces
        self.action_space_satellite = spaces.Discrete(3)
        self.action_space_debris = spaces.Discrete(3)
        self.observation_space_satellite = spaces.Box(self.low_satellite, self.high_satellite, dtype=np.int)
        self.observation_space_debris = spaces.Box(self.low_debris, self.high_debris, dtype=np.int)
   
    # Resets the environment
    def reset(self):
        del self.game
        self.game = PyGameObjectAvoidance()
      
        # Makes a list of obstacles, 200-300pixels away from one another 
        newpos = 0
        for i in range(50):
            rand = random.randint(0,2)
            self.game.debris.append(Debris(window_width + newpos))
            newpos += np.random.randint(270,310)

    
        # Updates the obstacles so they can move
        for i in self.game.debris:
            i.update()
    
       # Returns the initial state of the environment, for both agents
        self.state_satellite = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space(), self.game.debris[1].position_in_quantised_space())
        self.state_debris = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space())
        return np.array(self.state_satellite), np.array(self.state_debris)
  
    # One-step function of the environment
    def step(self, actions):
        action = actions[0]
        action_d = actions[1]
  
        assert self.action_space_satellite.contains(action), "%r (%s) invalid"%(action, type(action))
        assert self.action_space_debris.contains(action_d), "%r (%s) invalid"%(action_d, type(action_d))
       
        # The agents observes current state
        obs_satellite = self.state_satellite
        obs_debris = self.state_debris
        reward_satellite = 0
        reward_debris = 0
        done = False

        # The agents takes the action
        self.game.satellite.action(action)
        self.game.satellite.took_action = True

        i = self.game.debris[0]
        if i.posx < 80 and self.game.debris[1].posx < 200:
            self.game.debris[1].action(action_d)
            self.game.debris[0].action(action_d)
        elif i.posx < 200:
            i.action(action_d)

        # Observe state at t+1
        obs_satellite = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space(), self.game.debris[1].position_in_quantised_space())
        obs_debris = (self.game.satellite.position_in_quantised_space(), self.game.debris[0].position_in_quantised_space())
   
        # For the nearest obstacle, check if it is out of the environment
        if i.out_of_window:
            self.game.debris.pop(self.game.debris.index(i))

        # Determine the reward 
        if i.had_collision(self.game.satellite):
            reward_satellite = -100
            reward_debris = 100
            self.game.satellite.is_alive = False
        elif self.game.satellite.not_collision(i):
            reward_satellite = 10
            reward_debris = -10
        elif self.game.satellite.satellite_aligned_w_obstacle(i):
            reward_satellite = -1
            reward_debris = 1
       
        # Updates scoreboard
        if self.game.satellite.is_alive and i.out_of_window:
            if not i.checked:                            
                self.game.satellite.objects_avoided += 1
                i.checked = True

        # If satellite is dead the environment needs to reset
        if not self.game.satellite.is_alive:
            done = True

        obs = obs_satellite, obs_debris
        reward = reward_satellite, reward_debris

        return obs, reward, done, {}

    # Render the environment
    def render(self, mode="human", close=False):
        self.game.view()