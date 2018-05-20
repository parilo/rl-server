import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import DQN
from snake_model_dense import SnakeModelDense
from snake_model_dqn_cnn import SnakeModelCNN
# from snake_model import SnakeModel

class SnakeDQN ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.sess = train_loop.sess

        observation_shapes = self.train_loop.observation_shapes
        learning_rate = 2.5e-4

        # critic = SnakeModel(input_shapes=critic_input_shapes, output_size=1)
        # critic = SnakeModelDense(
        #     input_shapes = observation_shapes,
        #     output_size = 3
        # )

        critic = SnakeModelCNN(
            input_shapes = observation_shapes,
            output_size = 3
        )

        for v in critic.variables():
            print('--- critic v: {}'.format(v.name))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # self.observation_for_act = [
        #     tf.placeholder (tf.float32, observation_for_act_shapes[0])
        # ]

        self.observation_for_act = self.train_loop.get_observations_for_act_placeholders()[0]

        (batch_rewards,
        batch_actions,
        batch_prev_states,
        batch_next_states,
        batch_not_terminator) = self.train_loop.get_train_batches()

        print ('--- observation_for_act {}'.format(self.observation_for_act))
        print ('--- batch_rewards {}'.format(batch_rewards))
        print ('--- batch_actions {}'.format(batch_actions))
        print ('--- batch_prev_states {}'.format(batch_prev_states))
        print ('--- batch_next_states {}'.format(batch_next_states))
        print ('--- batch_not_terminator {}'.format(batch_not_terminator))

        self.controller = DQN(
            act_observations = self.observation_for_act,
            train_rewards = batch_rewards,
            train_given_actions = batch_actions,
            train_prev_observations = batch_prev_states,
            train_next_observations = batch_next_states,
            train_next_observations_mask = batch_not_terminator,
            batch_size = train_loop.batch_size,
            model = critic,
            optimizer = optimizer,
            discount_rate = train_loop.discount_rate,
            target_network_update_rate = 1.0
        )

        self.act_count = 0

    def act (self, state):

        scores, actions = self.sess.run (
            [
                self.controller.action_scores,
                self.controller.predicted_actions
            ],
            {
                self.observation_for_act: np.expand_dims(np.array(state), 0)
            }
        )

        self.act_count += 1
        return scores[0].tolist(), actions[0].tolist()

    def act_batch (self, states):
        a = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(states)
            }
        )
        return a.tolist ()

    def get_loss_op (self):
        return self.controller.prediction_error

    def get_train_ops (self):
        return [
            self.controller.train_op,
            self.controller.target_network_update
        ]
