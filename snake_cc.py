import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import CriticControl
# from tf_rl.models import MLP
from snake_model import SnakeModel

class SnakeRL ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.graph = train_loop.graph
        self.sess = train_loop.sess
        # journalist = train_loop.logger

        num_actions = self.train_loop.num_actions;
        observation_size = self.train_loop.observation_size;
        observations_in_seq = 1;
        input_size = observation_size*observations_in_seq;
        learning_rate = 1e-4

        r = tf.nn.relu
        t = tf.nn.tanh

        # critic = MLP([input_size, num_actions], [256, 256, 256, 256, 256, 1],
        #             [r, r, r, t, t, tf.identity], scope='critic')
        #
        # self.actor = MLP([input_size,], [256, 256, 256, 256, 256, num_actions],
        #             [r, r, r, r, t, t], scope='actor')

        observation_shape = [None, observation_size]
        critic_input_shape = [input_size, num_actions]
        action_shape = [num_actions]

        critic = SnakeModel(input_shapes=[[8, 8, 5], [num_actions]], output_size=1)

        # critic = MLP(critic_input_shape, [512, 512, 512, 512, 1],
        #             [r, r, r, r, tf.identity], scope='critic')

        # self.actor = MLP([input_size,], [512, 512, 512, 512, num_actions],
        #             [r, r, r, r, t], scope='actor')

        # step 1
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, epsilon=1e-4)
        # step 2
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self.observation_for_act = tf.placeholder (tf.float32, (1, observation_size))

        self.controller = CriticControl(
            observation_shape = observation_shape,
            action_shape = action_shape,
            batch_size = train_loop.batch_size,
            # self.actor,
            critic = critic,
            optimizer = optimizer,
            discount_rate=0.99,
            # target_actor_update_rate=0.01,
            target_critic_update_rate=0.01,
            rewards = self.train_loop.dequeued_rewards,
            given_action = self.train_loop.dequeued_actions,
            observation = self.train_loop.dequeued_prev_states,
            observation_for_act = self.observation_for_act,
            next_observation = self.train_loop.dequeued_next_states,
            # next_observation_mask = tf.ones(self.train_loop.dequeued_rewards.get_shape (), tf.float32)
            next_observation_mask = self.train_loop.dequeued_not_terminator
        )

        self.act_count = 0

    def act (self, state):

        if self.act_count < 500000:
            a, q, t, _ = self.sess.run (
                [
                    self.controller.actor_val,
                    self.controller.value_given_action_for_act,
                    self.controller.t,
                    self.controller.act_count_increase
                ],
                {
                    self.controller.observation_for_act: np.array(state).reshape((1,-1))
                }
            )

        else:
            a, q, t = self.sess.run (
                [
                    self.controller.actor_val,
                    self.controller.value_given_action_for_act,
                    self.controller.t
                ],
                {
                    self.controller.observation_for_act: np.array(state).reshape((1,-1))
                }
            )

        self.act_count += 1
        return a[0].tolist(), q[0].tolist(), t

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
            # self.controller.actor_update,
            self.controller.critic_update,
            self.controller.target_critic_update
        ]
