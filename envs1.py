import gym
from gym import spaces
import numpy as np
import math

class Space1(gym.Env):
    def __init__(self, env_config):
        self.max_step = env_config['max_step']
        self.integrator = env_config['integrator']
        # action space: velocity change
        self.action_space = spaces.Box(low=-5,high=5,shape=(2,)) # acceleration (ax,ay)
        # observation space: (velocity, position, Earth, Mars)
        #self.observation_space = spaces.Box(low=-50, high=50, shape=(8,)) # (vx,vy, x,y, x1,y1, x2,y2)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(4,)) # (vx,vy, x,y)
        self.step_count = 0
        
        # Use AU as distance unit
        self.au = 1.4959787e8 # km
        
        self.G_km = 6.67259e-20 # km^3 kg^-1 s^-2
        # Use km/s as velosity unit
        self.G = self.G_km / self.au**2 # km au^2 kg^-1 s^-2
        
        self.M = 1.989e30 # kg, mass of sun
        self.m_earth = 5.972e24
        self.m_mars = 6.39e23
        self.Ms = np.array([self.M, self.m_earth, self.m_mars])
        
        self.r_earth = 1
        self.r_mars = 2.2794e8 / self.au
        
        self.year = 365.256 * 86400 # seconds per year
        self.w_earth = 2 * math.pi / self.year  # s^-1
        self.w_mars = self.w_earth / self.r_mars**1.5
        self.ang_earth = 0
        self.ang_mars = 0
        
        self.t = 0 # passed time in seconds
        self.reset()
    
    def reset(self):
        self.step_count = 0
        #self.ang_earth = 0
        #self.ang_mars = 0.5
        self.t = 0
        # v = (12.5, 28.2) can reach mars when earth gravity is off
#         self.state = np.array([12.5, 28.2,
#                                self.r_earth + 2e-4, 0,
#                                self.r_earth, 0,
#                                self.r_mars, 0])
        v0 = math.sqrt(self.G_km / self.au * self.M / self.r_earth)
        #dv0 = v0 * (math.sqrt(2*self.r_mars/(self.r_earth+self.r_mars))-1)
        self.ecc = (self.r_mars - self.r_earth)/(self.r_mars + self.r_earth)
        self.c2 = np.array([-(self.r_mars - self.r_earth), 0])
        
        
        self.state = np.array([0, v0, self.r_earth + 2e-4, 0])
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        self.step_count += 1
        if self.step_count==self.max_step:
            done = True
            
        v = self.state[0:2]
        x = self.state[2:4]
        #x_e = self.state[4:6]
        #x_m = self.state[6:8]
        
        d = np.linalg.norm
        
        # Compute gravity
        r_s = d(x) # Sun
        a_s = -self.M * self.G / r_s**3 * x # km s^-2
        
#         def get_gravity(x, x_e, x_m):
#             r_s = d(x) # Sun
#             a_s = -self.M * self.G / r_s**3 * x
#             r_e = d(x - x_e) # Earth
#             a_e = -self.m_earth * self.G / r_e**3 * (x - x_e)
#             r_m = d(x - x_m) # Mars
#             a_m = -self.m_mars * self.G / r_m**3 * (x - x_m)
#             return np.array([a_s, a_e, a_m])
        
#         a_g = get_gravity(x, x_e, x_m)
#         a_g[1:2,:] = 1e-20 # remove earth gravity
#         print(a_g)
#         a = np.sum(a_g, axis=0) + action
        a = a_s + action
        
        # Update State
        # Constant timestep
        dt = 1e3
        # Variable timestep according to orbit period
#         print(d(a_g, axis=1))
        #dt = 5e-2 * np.min((self.G_km * self.Ms / d(a_g, axis=1)**3)**(1/4))
        dt = 5e-2 * (self.G_km * self.M / d(a_s)**3)**(1/4)
#         print(dt)

#         self.ang_earth = self.ang_earth + self.w_earth * dt
#         if self.ang_earth > np.pi: self.ang_earth -= np.pi*2
#         x_e_new = np.array([math.cos(self.ang_earth),
#                             math.sin(self.ang_earth)]) * self.r_earth
#         self.ang_mars = self.ang_mars + self.w_mars * dt
#         if self.ang_earth > np.pi: self.ang_earth -= np.pi*2
#         x_m_new = np.array([math.cos(self.ang_mars),
#                             math.sin(self.ang_mars)]) * self.r_mars        
        
        if self.integrator == "Euler":
            v_new = v + a * dt
            x_new = x + (v + v_new) * dt / 2 /self.au  
        elif self.integrator == "Leapfrog":
            v_new_half = v + a * dt / 2
            x_new = x + v_new_half * dt / self.au
            
            r_s_new = d(x_new)
            a_new = -self.M * self.G / r_s_new**3 * x_new + action
#             a_new = np.sum(get_gravity(x_new, x_e_new, x_m_new),
#                            axis=0) + action
            v_new = v_new_half + a_new * dt / 2
        else:
            print("ERROR")
        
        self.t += dt
        self.state[0:2] = v_new
        self.state[2:4] = x_new
#         self.state[4:6] = x_e_new
#         self.state[6:8] = x_m_new
        
        # Reward
        reward = 0
        if d(x_new) < 1e-2:
            done = True
            reward = -10000
        if d(x_new) > 2:
            done = True
            reward = -10000
        else:
            dr = d(x_new - self.c2)
            cos_theta = (x_new[0]-self.c2[0])/dr
            dr0 = (self.r_mars+self.r_earth)/2*(1-self.ecc**2)/(1-self.ecc*cos_theta)
            reward = -abs(dr-dr0)
        
#         if d(x_new - x_e_new) < 4.25e-5: # earth radius
#             done = True
#             reward = -10000
#         if d(x_new - x_m_new) < 1e-4:
#             done = True
#             reward = 10000
        
        return self.state, reward, done, {}
