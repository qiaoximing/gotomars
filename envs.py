import gym
from gym import spaces
import numpy as np
import math

class Orbit(gym.Env):
    def __init__(self, env_config):
        #self.directions = (-1, 1, 0) #speed down, up, keep
        self.max_step = env_config['max_step']
        # action space: velocity change
        self.action_space = spaces.Box(low=-5,high=5,shape=(1,))
        # observation space: (velocity, radius)
        # self.observation_space = spaces.Tuple((spaces.Box(low=-50,high=50,shape=(1,)), spaces.Box(low=-23,high=23,shape=(1,))))
        # RLLib doesn't work well with Tuple?
        self.max_v = 50
        self.max_r = 50
        max_obs = max(self.max_v, self.max_r)
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, shape=(2,))
        self.step_count = 0
        self.G = 6.67259e-17 # N km^2 / kg^2
        self.M = 1.989e30 # kg
        # self.m = 100 # doesn't matter
        # Use AU as standard distance unit
        self.au = self.r_earth_km = 1.4959787e8 # km
        self.r_mars_km = 2.2794e8 # km
        self.r_earth = 1
        self.r_mars = self.r_mars_km / self.au
        self.reset()
        
    def reset(self):
        v_escape = 29.79 # km/s
        #r_earth = 1.496*10**8 # km
        self.state = np.array([v_escape, -1.0])
        #self.state = np.array([v_escape, -r_earth])
        self.step_count = 0
        return self.state
        
    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        self.step_count += 1
        if self.step_count == self.max_step:
            done = True
        
        v0, r0 = self.state # auto cast to ndarray
        v1 = v0 + action
        #a = 1/(2/abs(r0)-v1**2*1000/self.G/self.M) # km
        a = 1/(2/abs(r0)-v1**2*1000*self.au/self.G/self.M)
        dE = 1/2/abs(v0**2-v1**2) # no need of self.m
        if r0>0:
            r1 = -2*a+r0
        else:
            r1 = 2*a+r0
            
        # If r1 goes to small, something's wrong
        if abs(r1) < 1e-2: # diameter of Sun ~ 1.3927e6 km ~ 1e-2 AU
            done = True
            reward = -10000
            return self.state, reward, done, {}
        v2 = v1*r0/r1
        # verify the new state
        if abs(v2) > self.max_v or abs(r1) > self.max_r:
            done = True
            reward = -10000
            return self.state, reward, done, {}
        self.state = np.array([v2[0], r1[0]])
        
        if abs(abs(r1) - self.r_mars) < 1e-2: # diameter of Mars ~ 6.8e3 km ~ 0.5e-4 AU
            done = True
            reward = 100 - dE[0] # hit Mars with minimal energy cost
        else:
            reward = -100 * abs(abs(r1[0]) - self.r_mars) - dE[0] # minimize the abs distance to get some reward
        return self.state, reward, done, {}


env_config_default = {'max_step': 10,}
