import time
import torch
import numpy as np

import rl_utils
from base_env import GazeboEnv
from train_VPG_BaseWorld import ActorCritic

# Set the parameters for the implementation
actor_lr=1e-3
critic_lr=1e-3
num_episodes=1000
seed=0
max_ep=500
discount=0.999
filename="VPG_BaseWorld"
directory="./pytorch_models"
device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
save_model=False
load_model=True
seed=0

# Create the training environment
environment_dim=20
robot_dim=4
env=GazeboEnv(environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim=environment_dim+robot_dim
action_dim=2
max_action=1

# Create the agent (pass discount and device in correct order)
agent=ActorCritic(state_dim,action_dim,actor_lr,critic_lr,discount,device,filename,directory)
if load_model:
    try:
        agent.load()
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# test
iter_episodes=100
max_steps=500


for i in range(10):
    rl_utils.evaluate(env,agent,i,max_steps)