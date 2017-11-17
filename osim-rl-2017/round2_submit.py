#!/usr/bin/env python

#  Script location : /home/round2_submit.py

import opensim as osim
from osim.redis.client import Client
from osim.env import *
import numpy as np
import argparse
import os
import tensorflow as tf
import json

env = RunEnv(visualize=False)
client = Client()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('/home/model-1480000.ckpt.meta')
saver.restore(sess, "/home/model-1480000.ckpt")
obs = tf.get_default_graph().get_tensor_by_name("observation_for_act:0")
act = tf.get_default_graph().get_tensor_by_name("taking_action/actor_action:0")
num_actions = 18

def preprocess_obs (obs):

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


prev_observation = preprocess_obs(client.env_create())
next_obs_seq.append_obs (prev_observation)

while True:

    prev_obs_seq.append_obs (prev_observation)

    s = json.loads(json.dumps({'s': prev_obs_seq.get_flatten_obs_seq()}))['s']
    action = sess.run (act,{
        obs: [s]
    })[0]
    action = np.array(action)
    action = np.array(json.loads(json.dumps({'action': action.tolist()}))['action'])
    # action = (np.array(action) + np.random.normal(scale=0.02, size=num_actions))
    action[action > 1.0] = 1.0
    action[action < 0.0] = 0.0
    action = action.tolist()

    [next_observation, reward, done, info] = client.env_step(action)
    print(next_observation)

    next_observation = preprocess_obs(next_observation)
    next_obs_seq.append_obs (next_observation)

    prev_observation = next_observation

    if done:
        prev_action = [0] * num_actions
        prev_obs_seq.reset()
        next_obs_seq.reset()

        prev_observation = client.env_reset()

        if not prev_observation:
            break

        prev_observation = preprocess_obs(prev_observation)
        next_obs_seq.append_obs (prev_observation)

client.submit()
