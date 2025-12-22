import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import rl_utils
from base_env import GazeboEnv

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        # 连续动作需要输出两个量：均值 mu 和 标准差 sigma
        self.mu_head = nn.Linear(600, action_dim)
        self.sigma_head = nn.Linear(600, action_dim)

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        # mu 使用 tanh 限制在 [-1, 1]
        mu = torch.tanh(self.mu_head(s))
        # sigma 必须大于 0，使用 softplus 处理，并加一个极小值防止为 0
        sigma = F.softplus(self.sigma_head(s)) + 1e-5
        return mu, sigma
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, 1)
    
    def forward(self,s):
        s=F.relu(self.layer_1(s))
        s=F.relu(self.layer_2(s))
        return self.layer_3(s)

#vanilla policy gradient network, or ActorCritic network
class ActorCritic(object):
    def __init__(self,state_dim,action_dim,actor_lr,critic_lr,discount,device,filename,directory):

        #actor network
        self.actor=Actor(state_dim,action_dim).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)

        #critic network
        self.critic=Critic(state_dim).to(device)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

        #other parameters
        self.iter_count=0
        self.discount=discount
        self.device=device
        self.filename=filename
        self.directory=directory
        self.action_dim=action_dim
    
    def get_action(self,state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        mu, sigma = self.actor(state)
        # 构建正态分布
        dist = torch.distributions.Normal(mu, sigma)
        # 采样一个动作 (exploratory action)
        action = dist.sample()
        # 限制动作范围在 -1 到 1 之间 (可选，视环境而定)
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().data.numpy().flatten()
    
    def update(self,transition_dict):

        #获取数据 get data
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, self.action_dim).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        #计算梯度
        td_target=rewards+self.discount*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)
        # 获取当前状态下的分布参数
        mu, sigma = self.actor(states)
        # 构建正态分布
        dist = torch.distributions.Normal(mu, sigma)
        # 计算当前动作在该分布下的对数概率密度 (Log Probability Density)
        log_probs = dist.log_prob(actions)
        
        # 如果动作是多维的，通常需要把所有维度的 log_prob 加起来
        log_probs = log_probs.sum(dim=1, keepdim=True) 

        actor_loss=torch.mean(-log_probs*td_delta.detach())
        critic_loss=torch.mean(F.mse_loss(self.critic(states),td_target.detach()))

        #更新网络 update network
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save(self):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (self.directory, self.filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (self.directory, self.filename))

    def load(self):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (self.directory, self.filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (self.directory, self.filename))
        )
        
if __name__ =="__main__":
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
    save_model=True
    load_model=False
    seed=0

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

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

    # train
    iter_episodes=100
    max_steps=500
    iterations=10

    rl_utils.train_on_policy_agent(env,agent,iter_episodes,max_steps,iterations)