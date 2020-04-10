import numpy as np
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from envs import Orbit

# GPU setup
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

# Ray setup
ray.init(num_cpus=4, num_gpus=1)

# Trainer setup
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 4
config['num_gpus'] = 1
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['eager'] = True
config['eager_tracing'] = True
# config['output'] = "./output"
config['model']['fcnet_hiddens'] = [100, 100]
# config['num_cpus_per_worker'] = 0 # for notebook execution

config['env_config']['max_step'] = 5
agent = PPOTrainer(config, env=Orbit)

def play(env, policy):
    print("Playing: ")
    obs = env.reset()
    done = False
    prev_action = np.array([0])
    prev_reward = 0
    while not done:
#         action_t, _ = policy.model.base_model(np.array([obs]))
#         action = action_t.numpy()[0][0:1]
        action = policy.compute_single_action(np.array(obs), None, prev_action, prev_reward, explore=False)[0]
        print(obs, action)
        obs, R, done, _ = env.step(action)
        prev_action = action
        prev_reward = R
    print(obs, R)
    
for i in range(100):
    result = agent.train()
    print(pretty_print(result))
    play(Orbit({'max_step': 5}), agent.get_policy())