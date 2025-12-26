import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99
BATCH_SIZE = 4
REPLAY_SIZE = 10
EPSILON = 0.5
STEPS = 5


# simple DQN network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], 16), nn.ReLU(), nn.Linear(16, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# Experience tuple and buffer
Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# convert batch to tensors
def batch_to_tensors(batch):
    states = torch.tensor([e.state for e in batch], dtype=torch.float32)
    actions = torch.tensor([e.action for e in batch], dtype=torch.long)
    rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
    dones = torch.tensor([e.done for e in batch], dtype=torch.bool)
    next_states = torch.tensor([e.new_state for e in batch], dtype=torch.float32)
    return states, actions, rewards, dones, next_states


# Compute DQN loass
def calc_loss(batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch_to_tensors(batch)
    state_action_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
    expected_values = rewards + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_values)


# Minimal Agent
class Agent:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.state, _ = env.reset()

    def play_step(self, net, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            q_vals = net(state_v)
            action = int(torch.argmax(q_vals).item())
        new_state, reward, done, trunc, _ = self.env.step(action)
        self.buffer.append(
            Experience(self.state, action, reward, done or trunc, new_state)
        )
        self.state = new_state
        return reward, done or trunc


# setup environmetn and network
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

net = DQN(obs_shape, n_actions)
tgt_net = DQN(obs_shape, n_actions)
tgt_net.load_state_dict(net.state_dict())

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
optimizer = optim.Adam(net.parameters(), lr=1e-3)


for step in range(STEPS):
    reward, done = agent.play_step(net, EPSILON)

    # forward pass inspection
    state_v = torch.tensor(agent.state, dtype=torch.float32).unsqueeze(0)
    q_vals = net(state_v)
    print(f"step{step} - Q-values: {q_vals.data.numpy()[0]}")
    print(f"step{step} - chosen action {int(torch.argmax(q_vals).item())}")

    # Compute loss if we have enought experience
    if len(buffer) >= 2:
        batch = buffer.sample(2)
        loss = calc_loss(batch, net, tgt_net)
        print(f"step{step} - loss {loss.item(): .4f}")

    if done:
        agent.state, _ = env.reset()

print("Micro end to end smoke test finished")
env.close()
