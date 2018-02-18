import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import ContinuousDeepQ
from tf_rl.models import MLP

class OsimRL ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.graph = train_loop.graph
        self.sess = train_loop.sess
        journalist = train_loop.logger

        num_actions = self.train_loop.num_actions;
        observation_size = self.train_loop.observation_size;
        observations_in_seq = 1;
        input_size = observation_size*observations_in_seq;
        learning_rate = 1e-4

        r = tf.nn.relu
        t = tf.nn.tanh

        # critic = MLP([input_size, num_actions], [512, 512, 512, 512, 512, 512, 1],
        #             [r, r, r, r, r, t, tf.identity], scope='critic')
        #
        # self.actor = MLP([input_size,], [512, 512, 512, 512, 512, 512, num_actions],
        #             [r, r, r, r, r, t, tf.nn.sigmoid], scope='actor')

        critic = MLP([input_size, num_actions], [512, 512, 512, 512, 512, 1],
                    [r, r, r, r, t, tf.identity], scope='critic')

        self.actor = MLP([input_size,], [512, 512, 512, 512, 512, num_actions],
                    [r, r, r, r, t, tf.nn.sigmoid], scope='actor')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-5)

        self.controller = ContinuousDeepQ(
            input_size,
            num_actions,
            self.actor,
            critic,
            optimizer,
            self.sess,
            discount_rate=0.99,
            target_actor_update_rate=0.01,
            target_critic_update_rate=0.01,
            exploration_period=5000,
            max_experience=10000,
            store_every_nth=4,
            train_every_nth=4,
            summary_writer=journalist,
            rewards = self.train_loop.dequeued_rewards,
            given_action = self.train_loop.dequeued_actions,
            observation = self.train_loop.dequeued_prev_states,
            next_observation = self.train_loop.dequeued_next_states,
            next_observation_mask = tf.ones(self.train_loop.dequeued_rewards.get_shape (), tf.float32)
        )

    def act (self, state):
        [a] = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(state).reshape((1,-1))
            }
        )
        return a.tolist ()

    def act_batch (self, states):
        a = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(states)
            }
        )
        return a.tolist ()

    def get_loss_op (self):
        return self.controller.critic_error

    def get_train_ops (self):
        return [
            self.controller.actor_update,
            self.controller.critic_update,
            self.controller.update_all_targets
        ]
