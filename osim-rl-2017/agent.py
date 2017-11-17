#!/usr/bin/env python

import argparse
import numpy as np
import random
from osim.env.run import RunEnv
from rl_client import RLClient

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
args = parser.parse_args()
vis = args.visualize

class ExtRunEnv(RunEnv):

    def __init__(self, *args, **kwargs):
        super(ExtRunEnv, self).__init__(*args, **kwargs)

    def step (self, action):
        observation, reward, done, info = super(ExtRunEnv, self).step(action)
        self.total_reward += reward
        print ('{} {}'.format(reward, self.total_reward))
        observation = self.preprocess_obs (observation)
        return observation, reward, done, info

    def preprocess_obs (self, obs):

    #position of the pelvis (rotation, x, y)
    #0 1 2
    #velocity of the pelvis (rotation, x, y)
    #3 4 5
    #rotation of each ankle, knee and hip (6 values)
    #6 7 8 9 10 11
    #angular velocity of each ankle, knee and hip (6 values)
    #12 13 14 15 16 17
    #position of the center of mass (2 values)
    #18 19
    #velocity of the center of mass (2 values)
    #20 21
    #positions (x, y) of head, pelvis, torso, left and right toes, left and right talus (14 values)
    #22 23 24 25 26 27 28 29 30 31 32 33 34 35
    #strength of left and right psoas: 1 for difficulty < 2, otherwise a random normal variable with mean 1 and standard deviation 0.1 fixed for the entire simulation
    #36 37
    #next obstacle: x distance from the pelvis, y position of the center relative to the the ground, radius.
    #38 39 40

        obs = np.array (obs)
        x = obs [1]
        y = obs [2]
        a = obs [0]

        obs [1] = 0

        obs [6] -= a
        obs [7] -= a
        obs [8] -= a
        obs [9] -= a
        obs [10] -= a
        obs [11] -= a

        obs [18] -= x
        obs [22] -= x
        obs [23] -= y
        obs [24] = 0
        obs [25] = 0
        obs [26] -= x
        obs [27] -= y

        # possibly error, I though that this infirmation
        # already included in observation
        obs [28] = 0
        obs [29] = 0
        obs [30] = 0
        obs [31] = 0
        obs [32] = 0
        obs [33] = 0
        obs [34] = 0
        obs [35] = 0

        obs [38] /= 100.0
        return obs.tolist()

    def reset (self, *args, **kwargs):
        self.prev_reward = 0
        self.total_reward = 0
        return self.preprocess_obs (super(ExtRunEnv, self).reset(*args, **kwargs))

env = ExtRunEnv(visualize=True, max_obstacles = 10)

rl_client = RLClient ()

num_actions = 18

class ObservationSequence (object):

    def __init__ (self):
        self.reset()

    def append_obs (self, obs):
        self.obs_seq = np.concatenate((self.obs_seq[1:], np.array([obs])), axis=0)

    def get_flatten_obs_seq (self):
        obs = np.copy(self.obs_seq)
        obs[0] = obs[1] - obs[0]
        obs[1] = obs[2] - obs[1]
        return obs.reshape((-1)).tolist()

    def reset(self):
        self.obs_seq = np.zeros((3, 41))


prev_obs_seq = ObservationSequence ()
next_obs_seq = ObservationSequence ()

prev_observation = env.reset(difficulty = 2)
prev_action = [0.0] * num_actions
next_obs_seq.append_obs (prev_observation)

init_action = None
def reset_init_action ():
    global init_action
    init_action = np.round(np.random.uniform (0, 0.7, size=18)).tolist()
def get_init_action ():
    return init_action

init_i = 0
reset_init_action ()

for i in xrange(1000000000):

    # for vis == false agetns (not visualized)
    # I try to randomize it's start state with
    # random activations for 20 steps and then starting to control
    init_i += 1
    if init_i < 20 and not vis:
        if init_i == 10:
            reset_init_action ()
        action_received2 = get_init_action ()
        next_observation, reward, done, info = env.step(action_received2)

    else:

        prev_obs_seq.append_obs (prev_observation)

        action_received = rl_client.act (prev_obs_seq.get_flatten_obs_seq())
        action = (np.array(action_received) + np.random.normal(scale=0.02, size=num_actions))
        #.tolist()
        action[action > 1.0] = 1.0
        action[action < 0.0] = 0.0

        next_observation, reward, done, info = env.step(action.tolist())
        next_obs_seq.append_obs (next_observation)

        if not done:
            rl_client.store_exp (
                reward,
                action_received,
                prev_obs_seq.get_flatten_obs_seq(),
                next_obs_seq.get_flatten_obs_seq()
            )
            # if not vis and init_i > 300:
            #     done = True
        else:
           if init_i < 999:
               fall_reward = 0.0  # 0.0 fot step 1 and -1.0 for step 2 training
               rl_client.store_exp (
                   fall_reward,
                   action_received,
                   prev_obs_seq.get_flatten_obs_seq(),
                   next_obs_seq.get_flatten_obs_seq()
               )
               print('fall, reward {}'.format(fall_reward))

        prev_observation = next_observation
        prev_action = action_received

    if done:
        init_i = 0
        reset_init_action ()
        rand = random.uniform(0, 1)
        if rand <= 0.1:
            prev_observation = env.reset(difficulty = 0)
        else:
            prev_observation = env.reset(difficulty = 2)

        prev_action = [0] * num_actions
        prev_obs_seq.reset()
        next_obs_seq.reset()
        next_obs_seq.append_obs (prev_observation)
