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


class Space(gym.Env):
    def __init__(self, env_config):
        self.max_step = env_config['max_step']
        self.integrator = env_config['integrator']
        # action space: velocity change
        self.action_space = spaces.Box(low=-5,high=5,shape=(2,)) # acceleration (ax,ay)
        # observation space: (velocity, position, Earth, Mars)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(8,)) # (vx,vy, x,y, x1,y1, x2,y2)
        self.step_count = 0
        
        # Use AU as distance unit
        self.au = 1.4959787e8 # km
        
        self.G_km = 6.67259e-20 # km^3 kg^-1 s^-2
        # Use km/s as velosity unit
        self.G = self.G_km / self.au**2 # km au^2 kg^-1 s^-2
        
        self.M = 1.989e30 # kg, mass of sun
        self.m_earth = 5.972e24
        self.m_mars = 6.39e23
        
        self.r_earth = 1
        self.r_mars = 2.2794e8 / self.au
        
        self.w_earth = 0.986/24/60/60 # degree/s
        self.w_mars = self.w_earth*1.881
        self.ang_earth = -90
        self.ang_mars = -90
        
        self.t = 0
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.ang_earth = -90
        self.ang_mars = -90
        self.t = 0
        self.state = np.array([5, 0,
                               0, -self.r_earth, 
                               0, -self.r_earth, 
                               0, -self.r_mars])
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        self.step_count += 1
        if self.step_count==self.max_step:
            done = True
            
        v = self.state[0:2]
        x = self.state[2:4]
        x_e = self.state[4:6]
        x_m = self.state[6:8]
        
        # Compute gravity
        def get_gravity(x):
            r_s = np.linalg.norm(x) # Sun
            a_s = -self.M * self.G / r_s**3 * x
#         print(a_s)
#         r_e = np.linalg.norm(x-x_e) #Earth
#         a_e = -self.m_earth * self.G / self.au**2 / 1000 / r_e**3 * (x-x_e)
#         r_m = np.linalg.norm(x-x_m) # Mars
#         a_m = -self.m_mars * self.G / self.au**2 / 1000 / r_m**3 * (x-x_m)
            a = a_s #+ a_e + a_m
            return a
        
        a_gravity = get_gravity(x)
        a = a_gravity + action
        
        # Update State
        # Constant timestep
        dt = 1e5
        # Variable timestep: a^(-3/4) according to Kepler's third law
        dt = 50 / np.linalg.norm(a_gravity)**0.75
#         print(dt)
        if self.integrator == "Euler":
            v_new = v + a * dt
            x_new = x + (v + v_new) * dt / 2 /self.au  
        elif self.integrator == "Leapfrog":
            v_new_half = v + a * dt / 2
            x_new = x + v_new_half * dt / self.au
            a_new = get_gravity(x_new) + action
            v_new = v_new_half + a_new * dt / 2
        else:
            print("ERROR")
        
        self.ang_earth = (self.ang_earth + self.w_earth * dt) % 360
#         if self.ang_earth > 180:
#             self.ang_earth -= 360
        x1 = math.cos(self.ang_earth / 180 * math.pi) * self.r_earth
        y1 = math.sin(self.ang_earth / 180 * math.pi) * self.r_earth
        self.ang_mars = (self.ang_mars + self.w_mars * dt) % 360
#         if self.ang_mars > 180:
#             self.ang_mars -= 360
        x2 = math.cos(self.ang_mars / 180 * math.pi) * self.r_mars
        y2 = math.sin(self.ang_mars / 180 * math.pi) * self.r_mars
        
        self.t += dt
        self.state[0:2] = v_new
        self.state[2:4] = x_new
        self.state[4:8] = np.array([x1, y1, x2, y2])
        
        # Reward
        reward = 0
        if np.linalg.norm(x_new) < 1e-2:
            done = True
            reward = -10000
        if np.linalg.norm(x_new - np.array([x2,y2])) < 1e-4:
            done = True
            reward = 10000
        
        return self.state, reward, done, {}
    
    
    
    
env_config_default = {'max_step': 10,}
