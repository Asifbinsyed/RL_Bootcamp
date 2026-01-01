#!/usr/bin/env python3
"""
DQN Pong Training Script - Folder agnostic
Can be run from any directory as long as the project structure is maintained.
"""
import sys
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Add script directory to Python path to ensure imports work
# This makes the script folder-agnostic - can be run from anywhere
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gymnasium as gym
import ale_py
from lib import dqn_model
from lib import wrappers
import config

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,           # current state
    torch.LongTensor,           # actions
    torch.Tensor,               # rewards
    torch.BoolTensor,           # done || trunc
    torch.ByteTensor            # next state
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
           dones_t.to(device),  new_states_t.to(device)


def calc_loss(batch: tt.List[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device, gamma: float) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Pong Training")
    parser.add_argument("--mode", default="local", choices=["local", "cloud", "cloud_test", "custom"],
                        help="Configuration mode: 'local' for M1 Pro testing, 'cloud' for full GPU training, 'cloud_test' for cloud GPU test run (30min), 'custom' for manual")
    parser.add_argument("--dev", default=None, 
                        help="Device name (overrides config: cpu, cuda, mps). If not set, uses config default")
    parser.add_argument("--env", default=None,
                        help="Environment name (overrides config)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to train (overrides config)")
    parser.add_argument("--max-minutes", type=int, default=None,
                        help="Maximum training time in minutes (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    cfg = config.get_config(args.mode)
    
    # Override config with command line arguments if provided
    if args.dev is not None:
        cfg.device = args.dev
    if args.env is not None:
        cfg.env_name = args.env
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    if args.max_minutes is not None:
        cfg.max_time_minutes = args.max_minutes
    
    device = torch.device(cfg.device)
    
    # Set up output directories relative to script location (folder-agnostic)
    models_dir = SCRIPT_DIR / "models"
    runs_dir = SCRIPT_DIR / "runs"
    models_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Models will be saved to: {models_dir}")
    print(f"TensorBoard logs will be saved to: {runs_dir}")
    print(f"Using device: {device}")
    print(f"Configuration: replay_size={cfg.replay_size}, batch_size={cfg.batch_size}, "
          f"replay_start_size={cfg.replay_start_size}, epsilon_decay_frames={cfg.epsilon_decay_last_frame}")
    if cfg.max_frames:
        print(f"Max frames limit: {cfg.max_frames}")
    if cfg.max_time_minutes:
        print(f"Max time limit: {cfg.max_time_minutes} minutes")
    
    env = wrappers.make_env(cfg.env_name)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(log_dir=str(runs_dir), comment=cfg.tensorboard_comment)
    print(net)

    buffer = ExperienceBuffer(cfg.replay_size)
    agent = Agent(env, buffer)
    epsilon = cfg.epsilon_start

    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_time = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        
        # Check time limit
        if cfg.max_time_minutes:
            elapsed_minutes = (time.time() - start_time) / 60.0
            if elapsed_minutes >= cfg.max_time_minutes:
                print(f"\nTime limit reached: {elapsed_minutes:.1f} minutes. Stopping training.")
                break
        
        # Check frame limit
        if cfg.max_frames and frame_idx >= cfg.max_frames:
            print(f"\nFrame limit reached: {frame_idx} frames. Stopping training.")
            break
        
        # Update epsilon
        epsilon = max(cfg.epsilon_final, 
                     cfg.epsilon_start - frame_idx / cfg.epsilon_decay_last_frame)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            elapsed_time = (time.time() - start_time) / 60.0
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                  f"eps {epsilon:.2f}, speed {speed:.2f} f/s, time {elapsed_time:.1f}m")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                model_name = cfg.env_name.replace("/", "_") + f"-best_{m_reward:.0f}.dat"
                model_path = models_dir / model_name
                torch.save(net.state_dict(), model_path)
                print(f"Model saved to: {model_path}")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > cfg.mean_reward_bound:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < cfg.replay_start_size:
            continue
        if frame_idx % cfg.sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(cfg.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device, cfg.gamma)
        loss_t.backward()
        optimizer.step()
    writer.close()
    print(f"\nTraining completed. Total frames: {frame_idx}, Total time: {(time.time() - start_time)/60:.1f} minutes")