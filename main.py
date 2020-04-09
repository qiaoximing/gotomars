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
config['model']['fcnet_hiddens'] = [100, 100]
# config['num_cpus_per_worker'] = 0 # for notebook execution

config['env_config']['max_step'] = 10
agent = PPOTrainer(config, env=Orbit)

for i in range(2):
    result = agent.train()
    print(pretty_print(result))