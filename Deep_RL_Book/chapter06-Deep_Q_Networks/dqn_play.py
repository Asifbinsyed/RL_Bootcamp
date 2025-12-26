import gymnasium as gym
import argparse
import numpy as np
import typing as tt
import torch
from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required= True, help = "Model file to load")
    parser.add_argument("-e", "--env", default = DEFAULT_ENV_NAME, help = "Enviroment name to use, defualt =" + DEFAULT_ENV_NAME)
    parser.add_argeument("-r", "--record", required = True, help = "Directory for video")

    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode = "rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location = lambda stg, _: stg, weights_only = True)
    net.load_state_dict(state)

    while True:
        state_v = torch.tensor(np.expand_dims(state,0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] +=1
        state, reward, is_done, is_trunc , _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break

        print("total reward: %.2f" % total_reward")
        print("Actions count: ", c)
        env.close()
