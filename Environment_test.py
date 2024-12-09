import numpy as np
import gym
from Environment import PhysicalEnv


Q = 2500.0
l_0 = 0.78
E = 2.08 * 1e11
A = np.pi / 4 * (0.004 ** 2)
xs = np.array([0.0, 0.04, 0.08, 0.28, 0.4])
x_c = 0.2
size = 5
max_steps = 3

gym.envs.register(
     id='Physical_Env-v0',
     entry_point='Environment:PhysicalEnv',
     max_episode_steps=3,
     kwargs={'Q' : Q, 'l_0' : l_0, 'E' : E, 'A' : A, 'xs' : xs, 'x_c' : x_c, 'size' : size, 'max_steps' : max_steps},
)
physical_env = gym.make('Physical_Env-v0')
#physical_env = PhysicalEnv(Q, l_0, E, A, xs, x_c, size)

forces, info = physical_env.reset()
l = info['l']
print(forces)
print(l)

action = {'index' : 3, 'u' : 0.0000268}
#action = physical_env.action_space.sample()
print(action)
forces, reward, terminated, truncated, info = physical_env.step(action)
l = info['l']
print(forces)
print(l)
print(reward)
print(terminated)
print(truncated)

action = {'index' : 0, 'u' : 0.0000095}
print(action)
forces, reward, terminated, truncated, info = physical_env.step(action)
l = info['l']
print(forces)
print(l)
print(reward)
print(terminated)
print(truncated)

action = {'index' : 2, 'u' : -0.000009}
print(action)
forces, reward, terminated, truncated, info = physical_env.step(action)
l = info['l']
print(forces)
print(l)
print(reward)
print(terminated)
print(truncated)
