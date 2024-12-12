import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy.optimize import fsolve

np.bool8 = np.bool_

class PhysicalEnv(gym.Env):

    def __init__(self, Q, l_0, E, A, xs, x_c, size = 5, max_steps = 3):
        assert(len(xs) == size)
        assert(xs[0] == 0)
        self.Q = Q
        self.l_0 = l_0
        self.E = E
        self.A = A
        self.xs = xs
        self.x_c = x_c
        self.size = size
        self.max_steps = max_steps
        self.EA = self.E * self.A
        self.l_0_EA = self.l_0 / self.EA
        self.action_scale = 1e4
        self.observation_space = spaces.Box(low = -np.inf * np.ones(self.size), high = np.inf * np.ones(self.size), shape=(self.size,), dtype=np.float64)

        self.action_space = spaces.Dict(
            {
                "index": spaces.Discrete(self.size),
                "u": spaces.Box(low = np.array([-np.inf]), high = np.array([np.inf]), shape=(1,), dtype=np.float64),
            }
        )
        
    def step(self, action):
        self.n_step = self.n_step + 1
        terminated = (self.n_step == self.max_steps)
        truncated = terminated
       
        line = np.array([1.0, 1.0])
        input_ = np.concatenate((self.f, line))
        solution = fsolve(self.step_equations_function, input_, args=action)
        assert np.allclose(self.step_equations_function(solution, action), np.zeros(self.size + 2))
        self.f = solution[:self.size]
        line = solution[self.size:]
        a = line[0]
        b = line[1]
        
        observation = self.f
        self.l = a * self.xs + b * np.ones(self.size)
        reward = 0.0
        if terminated:
            reward = np.min(self.f) - np.max(self.f)
        info = {'l' : self.l}


        return observation, reward, terminated, truncated, info
        
        
    def step_equations_function(self, args, action):
        f = args[:self.size]
        line = args[self.size:]
        index = action["index"]
        u = action["u"][0] / self.action_scale
        alpha = line[0]
        beta = line[1]
        result = []
        result.append(np.sum(f) - self.Q)
        result.append(np.dot(f, self.xs) - self.Q * self.x_c)
        for i in range(self.size):
            equation = self.l[i] * (1 + (f[i] - self.f[i]) / self.EA) - alpha * self.xs[i] - beta
            if i == index:
                equation = equation - u
            result.append(equation)
        
        return np.array(result)
        
    def reset_equations_function(self, f):
        result = []
        result.append(np.sum(f) - self.Q)
        result.append(np.dot(f, self.xs) - self.Q * self.x_c)
        for i  in range(self.size - 2):
            result.append((f[i+1] - f[0]) / self.xs[i+1] - (f[i+2] - f[0]) / self.xs[i+2])
        return np.array(result)
    
    def reset_equations_simple_function(self, line):
        a = line[0]
        b = line[1]
        f = a * self.xs + np.ones(self.size) * b
        result = []
        result.append(np.sum(f) - self.Q)
        result.append(np.dot(f, self.xs) - self.Q * self.x_c)
        return np.array(result)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        f = fsolve(self.reset_equations_function, np.ones(self.size))
        assert np.allclose(self.reset_equations_function(f), np.zeros(self.size))
        #line = fsolve(self.reset_equations_simple_function, np.ones(2))
        #assert np.allclose(self.reset_equations_simple_function(line), np.zeros(2))
        #a = line[0]
        #b = line[1]
        #f = a * self.xs + np.ones(self.size) * b
        
        self.f = f
        dl = self.f * self.l_0_EA
        self.l = np.ones(self.size) * self.l_0 + dl
        self.n_step = 0
            
        return self.f, {'l' : self.l}
            
            
