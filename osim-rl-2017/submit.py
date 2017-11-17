import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse
import tensorflow as tf

# history of my submits

# 5 / 79 711.515570
# 5 / 84 632.158343
# 5 / 85 1168.364093 6.212970
# 5 / 178 5.95614
# 5 / 177 7.276464
# 5 / 152 11.205569
# 5 / 151 13.749336

# 6 / 60 10.506293
# 6 / 165 17.011867
# 6 / 161 5.992715
# 6 / 184 5.377833
# 6 / 162 10.805526
# 6 / 164 15.496263

# 7 / 1 21.697849
# 7 / 1 12.186219
# 7 / 10 16.194243
# 7 / 51 11.748456
# 7 / 52 16.956996
# 7 / 53 15.723823
# 7 / 55 23.645475
# 7 / 56 23.282150
# 7 / 57 22.796935

# 8 / 58

# 9 / 20 22.915546
# 9 / 21 22.805862

# 10 / 230 18.730473
# 10 / 372 18.630634
# 10 / 374 11.940716

# 11 / 220 21.9924
# 11 / 277 19.000357

# 12 / 100 4.4816
# 12 / 216 18.409287
# 12 / 217 28.958674
# 12 / 218 timeout

# 17 / 24 14.946525
# 17 / 24 16.793447
# 17 / 33 2.1
# 17 / 32 server error bad gateway
# 17 / 34 28.097300 one quick fall

# 20 / 50 24.926265 last one quick fall
# 20 / 63 4.715645
# 20 / 85 23.793841
# 20 / 84 27.466175
# 20 / 83 38.012350
# 20 / 99 26.965255
# 20 / 100 25.597062
# 20 / 101 26.844620
# 20 / 98 38.393103
# 20 / 97 38.549363 16.776787
# 20 / 96 14.279153
# 20 / 85 27.066775

# 22 / 194 37.611422
# 22 / 196 37.723519
# 22 / 424 37.548924

# 23 / 65  37.966370
# 23 / 356 36.899593

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('submissions/25/model-1250000.ckpt.meta')
saver.restore(sess, "submissions/25/model-1250000.ckpt")
obs = tf.get_default_graph().get_tensor_by_name("observation_for_act:0")
act = tf.get_default_graph().get_tensor_by_name("taking_action/actor_action:0")

# Settings
remote_base = 'http://grader.crowdai.org:1729'

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

class ObservationSequence (object):

    def __init__ (self):
        self.obs_seq = np.zeros((3, 41))

    def append_obs (self, obs):
        self.obs_seq = np.concatenate((self.obs_seq[1:], np.array([obs])), axis=0)

    def get_flatten_obs_seq (self):
        return self.obs_seq.reshape((-1)).tolist()

prev_obs_seq = ObservationSequence ()
next_obs_seq = ObservationSequence ()

# Create environment
client = Client(remote_base)
prev_observation = preprocess_obs(client.env_create('place your key here'))
next_obs_seq.append_obs (prev_observation)

total_reward = 0.0
iteration = 0
while True:

    prev_obs_seq.append_obs (prev_observation)

    action = sess.run (act,{
        obs: [prev_obs_seq.get_flatten_obs_seq()]
    })[0]
    [next_observation, reward, done, info] = client.env_step(action.tolist())
    iteration += 1

    total_reward += reward
    print('total reward: {} {} {}'.format(iteration, total_reward, next_observation[2]))

    next_observation = preprocess_obs(next_observation)
    next_obs_seq.append_obs (next_observation)

    prev_observation = next_observation

    if done:
        print('---------------------')
        print('--- done episode ----')
        print('---------------------')
        prev_observation = client.env_reset()
        iteration = 0
        if not prev_observation:
            break
        prev_observation = preprocess_obs(prev_observation)

client.submit()
