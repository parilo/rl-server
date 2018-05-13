import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import CriticControl
from tf_rl.models import MLP
# from snake_model import SnakeModel

class SnakeRL ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.graph = train_loop.graph
        self.sess = train_loop.sess

        num_actions = self.train_loop.num_actions
        num_agents_per_client = 1
        observation_shapes = self.train_loop.observation_shapes
        # observations_in_seq = 1;
        # input_size = observation_size*observations_in_seq;
        learning_rate = 1e-4

        observation_for_act_shapes = []
        for shape in train_loop.observation_shapes:
            observation_for_act_shapes.append(
                tuple([num_agents_per_client] + list(shape))
            )

        action_shape = [num_actions]
        critic_input_shapes = list(observation_shapes)
        critic_input_shapes.append(action_shape)


        r = tf.nn.relu
        t = tf.nn.tanh
        critic = MLP([observation_shapes[0][0], num_actions], [256, 256, 256, 256, 256, 1],
                    [r, r, r, t, t, tf.identity], scope='critic')
        # critic = MLP([observation_shapes[0][0], num_actions], [512, 512, 512, 512, 1],
        #             [r, r, r, r, tf.identity], scope='critic')

        # critic = SnakeModel(input_shapes=critic_input_shapes, output_size=1)


        for v in critic.variables():
            print('--- critic v: {}'.format(v.name))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, epsilon=1e-4)

        self.observation_for_act = [
            tf.placeholder (tf.float32, observation_for_act_shapes[0])
        ]

        self.controller = CriticControl(
            observation_shape = train_loop.observation_shapes,
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

        # if self.act_count < 500000:
        #     a, q, t = self.sess.run (
        #         [
        #             self.controller.actor_val,
        #             self.controller.value_given_action_for_act,
        #             self.controller.t
        #         ],
        #         {
        #             self.controller.observation_for_act: np.array(state)
        #         }
        #     )
        #
        # else:
        a, q, t = self.sess.run (
            [
                self.controller.actor_val,
                self.controller.value_given_action_for_act,
                self.controller.t
            ],
            {
                self.controller.observation_for_act[0]: np.expand_dims(np.array(state), 0)
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
