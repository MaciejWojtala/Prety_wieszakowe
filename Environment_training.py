from pprint import pprint
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import gymnasium as gym
from Environment import PhysicalEnv

from ray.tune.registry import register_env

save_path = "/home/maciek/Pulpit/Research/RL/Prety_wieszakowe/results"
policy_path = "/home/maciek/Pulpit/Research/RL/Prety_wieszakowe/policy_results"
checkpoint_path = save_path + "/learner_group/learner/rl_module/default_policy"
N_EPOCHS = 1000
EVAL_ITERATIONS = 1000

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

def env_creator(env_config):
    return PhysicalEnv(Q, l_0, E, A, xs, x_c, size, max_steps)  # return an env instance

register_env("Physical_Env-v0", env_creator)

def train():
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Physical_Env-v0")
        .env_runners(num_env_runners=1)
    )

    algo = config.build()

    for i in range(N_EPOCHS):
        result = algo.train()
        result.pop("config")

        if i % 100 == 0 or i == N_EPOCHS - 1:
            pprint(result)
            checkpoint_dir = algo.save_to_path(path = save_path)
            print(f"Checkpoint saved in directory {checkpoint_dir}")

def test():
    algo = Algorithm.from_checkpoint(save_path)
    loaded_module = RLModule.from_checkpoint(checkpoint_path)
    action_dist_class = loaded_module.get_inference_action_dist_cls()
    best_reward = None
    best_forces = None
    best_actions = None
    for _ in range(EVAL_ITERATIONS):
        reward = None
        forces = []
        actions = []
        obs, info  = physical_env.reset()
        forces.append(obs)
        for i in range(max_steps):
            fwd_outputs = loaded_module.forward_exploration({"obs": torch.from_numpy(obs).float()})
            action_dist = action_dist_class.from_logits(
                fwd_outputs["action_dist_inputs"]
            )
            action = action_dist.sample()
            actions.append(action)
            obs, reward, terminated, truncated, info = physical_env.step(action)
            forces.append(obs)
        if best_reward is None or best_reward < reward:
            best_reward = reward
            best_forces = forces
            best_actions = actions
    print("reward")
    print(best_reward)
    print("forces")
    print(best_forces[0])
    for i in range(max_steps):
        print("action")
        print(best_actions[i])
        print("forces")
        print(best_forces[i+1])
    
    #result = algo.evaluate()
    #print(result)
        

train()
test()

