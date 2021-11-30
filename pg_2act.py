'''ref : https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/pendulum_a2c_online.ipynb'''

import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)


# helper function to convert numpy arrays to tensors
def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class Actor_cate(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)

class Actor_cont(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions)
        )

        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)

        return torch.distributions.Normal(means, stds)


## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )

    def forward(self, X):
        return self.model(X)


def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions1 = []
    actions2 = []
    states = []
    next_states = []
    rewards = []
    dones = []
    for action1, action2, reward, state, next_state, done in memory:
        actions1.append(action1)
        actions2.append(action2)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)

    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions1 = t(actions1).view(-1, 1)
    actions2 = t(actions2).view(-1, 1)
    states = t(states)
    next_states = t(next_states)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)
    return actions1, actions2, rewards, states, next_states, dones


def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


class A2CLearner():
    def __init__(self, actor1, actor2, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor1 = actor1
        self.actor2 = actor2
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor1_optim = torch.optim.Adam(actor1.parameters(), lr=actor_lr)
        self.actor2_optim = torch.optim.Adam(actor2.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    def learn(self, memory, steps, discount_rewards=True):
        actions1, actions2, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards)
        # actor 1
        if discount_rewards:
            td_target = rewards[-1]
        else:
            td_target = rewards[-1] + self.gamma * self.critic(next_states[-1]) * (1 - dones[-1])
        value = self.critic(states[-1])
        advantage = td_target - value

        probs = self.actor1(t(states[-1]))
        dist = torch.distributions.Categorical(probs=probs)
        # action = dist.sample()
        actor_loss1 = -dist.log_prob(actions1[-1]) * advantage.detach()
        self.actor1_optim.zero_grad()
        actor_loss1.backward()
        clip_grad_norm_(self.actor1_optim, self.max_grad_norm)
        writer.add_histogram("gradients/actor1",
                             torch.cat([p.grad.view(-1) for p in self.actor1.parameters()]), global_step=steps)
        writer.add_histogram("parameters/actor1",
                             torch.cat([p.data.view(-1) for p in self.actor1.parameters()]), global_step=steps)
        self.actor1_optim.step()

        # actor 2
        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        value = self.critic(states)
        advantage = td_target - value

        norm_dists = self.actor2(states)
        logs_probs = norm_dists.log_prob(actions2)
        entropy = norm_dists.entropy().mean()
        actor_loss2 = (-logs_probs * advantage.detach()).mean() - entropy * self.entropy_beta
        self.actor2_optim.zero_grad()
        actor_loss2.backward()
        clip_grad_norm_(self.actor2_optim, self.max_grad_norm)
        writer.add_histogram("gradients/actor2",
                             torch.cat([p.grad.view(-1) for p in self.actor2.parameters()]), global_step=steps)
        writer.add_histogram("parameters/actor2",
                             torch.cat([p.data.view(-1) for p in self.actor2.parameters()]), global_step=steps)
        self.actor2_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        writer.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        writer.add_histogram("parameters/critic",
                             torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()

        # reports
        writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
        writer.add_scalar("losses/entropy", entropy, global_step=steps)
        writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps)
        writer.add_scalar("losses/actor1", actor_loss1, global_step=steps)
        writer.add_scalar("losses/actor2", actor_loss2, global_step=steps)
        writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        writer.add_scalar("losses/critic", critic_loss, global_step=steps)

'''
class Runner():
    def __init__(self, env):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []

    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()

    def run(self, max_steps, memory=None):
        if not memory: memory = []

        for i in range(max_steps):
            if self.done: self.reset()

            dists = actor(t(self.state))
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low.min(), env.action_space.high.max())

            next_state, reward, self.done, info = self.env.step(actions_clipped)
            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
                writer.add_scalar("episode_reward", self.episode_reward, global_step=self.steps)

        return memory



env = gym.make("Pendulum-v0")
'''
writer = SummaryWriter("runs/mish_activation")
'''
# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
actor = Actor(state_dim, n_actions, activation=Mish)
critic = Critic(state_dim, activation=Mish)

learner = A2CLearner(actor, critic)
runner = Runner(env)


steps_on_memory = 16
episodes = 500
episode_length = 200
total_steps = (episode_length*episodes)//steps_on_memory

for i in range(total_steps):
    memory = runner.run(steps_on_memory)
    learner.learn(memory, runner.steps, discount_rewards=False)
'''

