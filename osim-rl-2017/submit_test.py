from osim.env.run import RunEnv
import numpy as np
import argparse
import tensorflow as tf
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('submissions/32/model-660000.ckpt.meta')
saver.restore(sess, "submissions/32/model-660000.ckpt")
obs = tf.get_default_graph().get_tensor_by_name("observation_for_act:0")
act = tf.get_default_graph().get_tensor_by_name("taking_action/actor_action:0")
num_actions = 18

#[[random.randint(0,2), random.randint(0, 10000000)] for i in range(30)]
check_envs = [[2, 11], [2, 22], [2, 33], [2, 9752825], [1, 4274759],
[2, 5050684], [0, 9922769], [2, 8623668], [0, 3592727], [2, 8463032],
[2, 5427207], [1, 585993], [0, 9759655], [0, 491740], [1, 3540735],
[0, 4602781], [2, 8745680], [1, 8694421], [2, 2651108], [2, 5976496],
[0, 6947040], [1, 3560257], [1, 1919437], [0, 2150806], [1, 4077707],
[2, 1782359], [2, 1630894], [2, 3793576], [1, 5219981], [1, 7695497]]

def preprocess_obs (obs):

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
    #obs /= 10.0
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

env = RunEnv(visualize=False, max_obstacles = 10)

class ObservationSequence (object):

    def __init__ (self):
        self.reset()

    def append_obs (self, obs):
        self.obs_seq = np.concatenate((self.obs_seq[1:], np.array([obs])), axis=0)

    def append_act (self, act):
        # self.acts_seq = np.concatenate((self.acts_seq[1:], np.array([act])), axis=0)
        pass

    def get_flatten_obs_seq (self):
        obs = np.copy(self.obs_seq)
        obs[0] = obs[1] - obs[0]
        obs[1] = obs[2] - obs[1]
        return obs.reshape((-1)).tolist()
        # return self.obs_seq.reshape((-1)).tolist()

    def reset(self):
        self.obs_seq = np.zeros((3, 41))


prev_obs_seq = ObservationSequence ()
next_obs_seq = ObservationSequence ()

def process_act (act):
    action = np.array (act)
    # action /= 10.0
    return action.tolist ()

def test_env(difficulty, seed):

    prev_observation = preprocess_obs(env.reset(difficulty = difficulty, seed = seed))
    next_obs_seq.append_obs (prev_observation)

    total_reward = 0.0
    # Run a single step
    #
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    iteration = 0
    while True:

        prev_obs_seq.append_obs (prev_observation)

        s = json.loads(json.dumps({'s': prev_obs_seq.get_flatten_obs_seq()}))['s']
        action = sess.run (act,{
            obs: [s]
        })[0]
        action = np.array(action)
        action = np.array(json.loads(json.dumps({'action': action.tolist()}))['action'])
        # action = (action + np.random.normal(scale=0.02, size=num_actions))
        action[action > 1.0] = 1.0
        action[action < 0.0] = 0.0
        action = process_act(action)
        [next_observation, reward, done, info] = env.step(action)

        total_reward += reward
        print('total reward: {} {} {}'.format(iteration, total_reward, next_observation[2]))

        next_observation = preprocess_obs(next_observation)
        next_obs_seq.append_obs (next_observation)

        prev_observation = next_observation
        iteration += 1

        if done:
            prev_action = [0] * num_actions
            prev_obs_seq.reset()
            next_obs_seq.reset()

            return total_reward

all_total_reward = 0.0
for i, env_params in zip(range(len(check_envs)), check_envs):
    all_total_reward += test_env(2, env_params[1])
    print '----------'
    print '--- ep: {} {}'.format(i, all_total_reward)
    print '----------'

print 'reward: {}'.format(all_total_reward / len(check_envs))
