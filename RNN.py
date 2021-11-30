'''ref : https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/pendulum_a2c_online.ipynb'''

import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from gym import spaces
from torch.autograd import Variable
from common_reader import MLP, Memory

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)


# helper function to convert numpy arrays to tensors
def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

class RNN(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            #PrintLayer(),
            nn.Linear(state_dim, 64),
            #PrintLayer(),
            activation(),
            # PrintLayer(),
            nn.Linear(64, 64),
            #PrintLayer(),
            activation(),
            #PrintLayer(),
            nn.Linear(64, n_actions),
            #PrintLayer()  # Add Print layer for debug
            nn.Softmax(dim=0)
        )
        self.rnn = nn.RNN(input_size=state_dim, hidden_size=n_hidden, dropout=0.3)
        self.W = nn.Parameter(torch.randn([n_hidden, n_actions]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_actions]).type(dtype))
        self.Softmax = nn.Softmax(dim=0)
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
        self.replay_memory = Memory(4500)

    def forward(self, X):
        #print(X)
        return torch.distributions.Categorical(torch.clamp(self.model(X), min = 1e-3))
        #action= torch.multinomial(self.model(X),1)
        #return action.item()
        #means = self.model(X)
        #stds = torch.clamp(self.logstds.exp(), 1e-3)

        #return torch.distributions.Normal(means, stds)


## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1)
            #PrintLayer(),
            #nn.Softmax(dim=0)
            #PrintLayer()
        )

    def forward(self, X):
        #return torch.distributions.Categorical(torch.clamp(self.model(X), min = 1e-3))
        return self.model(X)


def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    tags = []
    tag_probs = []

    for action, reward, state, next_state, done, tag, tag_prob in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
        tags.append(tag)
        tag_probs.append(tag_prob)

    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).view(-1, 1)
    states = t(states)
    next_states = t(next_states)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)
    tags = t(tags).view(-1,1)
    tag_probs = t(tag_probs).view(-1,1)

    return actions, rewards, states, next_states, dones, tags, tag_probs


def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


class A2CLearner():
    def __init__(self, actor_tag, actor_action, critic, gamma=0.9, entropy_beta=0, use_cuda=False,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5,batch_size = 32, random_trial_num = 4000, memory_capacity = 4500):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor_tag = actor_tag
        self.actor_action = actor_action
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_tag_optim = torch.optim.Adam(actor_tag.parameters(), lr=actor_lr)
        self.actor_action_optim = torch.optim.Adam(actor_action.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        #self.Session_block = Session_block
        self.batch_size = batch_size
        self.gamma = gamma
        #self.tau = tau
        self.tau_offset = 0
        self.replay_memory = Memory(memory_capacity)
        self.random_trial_num = random_trial_num

    def learn(self, state, action, next_state, reward, done, tag, tag_prob, steps, discount_rewards=True):
        #actions, rewards, states, next_states, dones, tag, tag_prob = process_memory(memory, self.gamma, discount_rewards)
        self.replay_memory.add_event(Memory.Event(state.copy(), action, next_state.copy(), reward, done, tag, tag_prob))
        # sample from replay memory
        if self.random_trial_num <= len(self.replay_memory.mem):  # start update when enough memories are gathered
            mini_batch = self.replay_memory.sample(self.batch_size)
            mini_batch = Memory.Event(*zip(*mini_batch))  # do this for batch processing

            #loss = nn.CrossEntropyLoss()
            #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #tag_vec = torch.squeeze(Variable(self.Tensor(mini_batch.tag)),1)
            #tag_vec = Variable(self.Tensor(mini_batch.tag))
            #tag_vec = torch.zeros(len(mini_batch.tag),4)
            #tag_vec[torch.arange(len(mini_batch.tag)),mini_batch.tag] = 1
            td_target_value = self.critic(Variable(self.Tensor(mini_batch.next_state)))
            #td_target_value = cos(self.critic(Variable(self.Tensor(mini_batch.next_state))).probs,tag_vec.long())
            if discount_rewards:
                td_target = torch.transpose(torch.unsqueeze(Variable(self.Tensor(mini_batch.reward)),0),0,1)
            else:
                #td_target_value = torch.distributions.Categorical(self.critic.model(Variable(self.Tensor(mini_batch.next_state)))).probs
                #td_target_value = self.critic(Variable(self.Tensor(mini_batch.next_state))).probs
                td_target = torch.transpose(torch.unsqueeze(Variable(self.Tensor(mini_batch.reward)),0),0,1) + \
                            self.gamma * td_target_value * (1 - torch.transpose(torch.unsqueeze(Variable(self.Tensor(mini_batch.done)),0),0,1))
            value = self.critic(Variable(self.Tensor(mini_batch.state)))
            advantage = td_target - value

            # actor
            policy_tag = self.actor_tag(Variable(self.Tensor(mini_batch.state)))
            # logs_probs = norm_dists.log_prob(actions)
            logs_pol_tag = policy_tag.log_prob(Variable(self.Tensor(mini_batch.tag)))
            entropy_tag = policy_tag.entropy().mean()
            actor_tag_loss = (-logs_pol_tag * advantage.detach()).mean() - entropy_tag * self.entropy_beta
            self.actor_tag_optim.zero_grad()
            actor_tag_loss.backward()
            clip_grad_norm_(self.actor_tag_optim, self.max_grad_norm)
            self.actor_tag_optim.step()
            # policy = torch.distributions.Categorical(self.actor.model(Variable(self.Tensor(mini_batch.state))))
            policy_action = self.actor_action(Variable(self.Tensor(mini_batch.state)))
            #logs_probs = norm_dists.log_prob(actions)
            logs_pol_action = policy_action.log_prob(Variable(self.Tensor(mini_batch.action)))
            entropy_action = policy_action.entropy().mean()
            actor_action_loss = (-logs_pol_action * advantage.detach()).mean() - entropy_action * self.entropy_beta
            self.actor_action_optim.zero_grad()
            #print('training input:')
            #print(Variable(self.Tensor(mini_batch.next_state)))
            #print('critic_model output:')
            #print(self.critic.model(Variable(self.Tensor(mini_batch.next_state))))
            #print('critic_model category:')
            #print(torch.distributions.Categorical(self.critic.model(Variable(self.Tensor(mini_batch.next_state)))))
            #print('critic_model category probs:')
            #print(torch.distributions.Categorical(self.critic.model(Variable(self.Tensor(mini_batch.next_state)))).probs)
            #print('td_target_value')
            #print(td_target_value)
            #print('tag vector')
            #print(tag_vec.long())
            #print('td_target_loss_part')
            #print(loss(td_target_value, tag_vec.long()))
            #print('td_target')
            #print(td_target)
            #print('value')
            #print(value)
            #print('logs_pol')
            #print(logs_pol)
            #print('advantage:')
            #print(advantage)
            #print('entropy')
            #print(entropy)
            #print('entropy_beta')
            #print(self.entropy_beta)
            #print("actor_loss : ")
            #print(actor_loss)
            actor_action_loss.backward()
            clip_grad_norm_(self.actor_action_optim, self.max_grad_norm)
            #writer.add_histogram("gradients/actor",
            #                     torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
            #writer.add_histogram("parameters/actor",
            #                     torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
            self.actor_action_optim.step()

            # critic
            critic_loss = F.mse_loss(td_target, value)
            #critic_loss = advantage
            self.critic_optim.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_optim, self.max_grad_norm)
            #writer.add_histogram("gradients/critic",
            #                     torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
            #writer.add_histogram("parameters/critic",
            #                     torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
            self.critic_optim.step()

            # reports
            writer.add_scalar("losses/log_tag_probs", -logs_pol_tag.mean(), global_step=steps)
            writer.add_scalar("losses/entropy_tag", entropy_tag, global_step=steps)
            writer.add_scalar("losses/log_action_probs", -logs_pol_action.mean(), global_step=steps)
            writer.add_scalar("losses/entropy_action", entropy_action, global_step=steps)
            writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps)
            writer.add_scalar("losses/actor_tag", actor_tag_loss, global_step=steps)
            writer.add_scalar("losses/actor_action", actor_action_loss, global_step=steps)
            writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
            writer.add_scalar("losses/critic", critic_loss, global_step=steps)
        else:
            advantage = torch.tensor([-1])
        #return advantage

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

