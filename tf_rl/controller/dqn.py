# -*- coding: utf-8 -*-

# DQN taken from
# https://github.com/nivwusquorum/tensorflow-deepq
# thank you Szymon Sidor ;)

import tensorflow as tf
import numpy as np

class DQN(object):
    # Описание параметров ниже
    def __init__(self,
            act_observations,
            train_rewards,
            train_given_actions,
            train_prev_observations,
            train_next_observations,
            train_next_observations_mask,
            batch_size,
            model,  # observation_to_actions,
            optimizer,
            discount_rate=0.95,
            target_network_update_rate=0.01):

        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_shapes : list of lists or tuple of tuples
            supported multiple input models
        num_actions : int
            number of actions that the model can execute
        model:
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, num_actions]
        optimizer:
            optimizer for prediction error
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        target_network_update_rate: float
            how much to update target network after each
            iteration. Let's call target_network_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        """

        self.act_observations = act_observations
        self.train_rewards = train_rewards
        self.train_given_actions = train_given_actions
        self.train_prev_observations = train_prev_observations
        self.train_next_observations = train_next_observations
        self.train_next_observations_mask = train_next_observations_mask
        self.batch_size = batch_size

        self.q_network = model
        self.optimizer = optimizer

        self.discount_rate             = tf.constant(discount_rate)
        self.target_network_update_rate = \
                tf.constant(target_network_update_rate)

        self.create_variables()

    def create_variables(self):
        self.target_q_network    = self.q_network.copy(scope="target_network")

        with tf.name_scope("taking_action"):
            # self.observation        = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
            self.action_scores = tf.identity(self.q_network(self.act_observations), name="action_scores")
            self.predicted_actions = tf.argmax(self.action_scores, axis=1, name="predicted_actions")

        with tf.name_scope("estimating_future_rewards"):
            self.next_action_scores = tf.stop_gradient(
                self.target_q_network(self.train_next_observations)
            )
            target_values = tf.identity(
                tf.reduce_max(
                    self.next_action_scores,
                    reduction_indices=[1,]
                ) * self.train_next_observations_mask,
                name="target_values"
            )
            self.future_rewards = tf.identity(
                self.train_rewards + self.discount_rate * target_values,
                name="future_rewards"
            )

        with tf.name_scope("q_value_precition"):

            scores = self.q_network(self.train_prev_observations)
            # print ('--- scores {}'.format(scores))
            # print ('--- train_given_actions {}'.format(self.train_given_actions))
            range_indices = tf.constant(np.expand_dims(np.arange(0, self.batch_size, dtype=np.int32), axis=1))
            actions_indices = tf.concat(
                [
                    range_indices,
                    tf.expand_dims(self.train_given_actions, axis=1)
                ],
                axis=1
            )
            # print ('--- actions_indices {}'.format(actions_indices))
            # print ('--- slice {}'.format(
            #     tf.gather_nd(
            #         scores,
            #         actions_indices
            #     )
            # ))

            self.train_actions_scores = tf.identity(
                tf.gather_nd(scores, actions_indices),
                name='train_actions_scores'
            )
            # self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,], name="masked_action_scores")



# ###################### Neural network architecture ######################
#
# input_shape = [None] + state_shape
# self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
#
# self.q_values = full_module(self.input_states, convs, fully_connected,
#                             num_actions, activation_fn)
#
# ######################### Optimization procedure ########################
#
# # one-hot encode actions to get q-values for state-action pairs
# self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
# actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
# q_values_selected = tf.reduce_sum(tf.multiply(self.q_values, actions_onehot), axis=1)
#
# # choose best actions (according to q-values)
# self.q_argmax = tf.argmax(self.q_values, axis=1)
#
# # create loss function and update rule
# self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
# self.td_error = tf.losses.huber_loss(self.targets, q_values_selected)
# self.loss = tf.reduce_sum(self.td_error)
# self.update_model = optimizer.minimize(self.loss)


            td_error = tf.losses.huber_loss(self.future_rewards, self.train_actions_scores)
            self.prediction_error = tf.reduce_sum(td_error)

            # td_error = tf.identity(self.train_actions_scores - self.future_rewards, name="td_error")
            # self.prediction_error = tf.reduce_mean(tf.square(td_error), name="prediction_error")

            gradients = self.optimizer.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # # Add histograms for gradients.
            # for grad, var in gradients:
            #     tf.histogram_summary(var.name, var)
            #     if grad is not None:
            #         tf.histogram_summary(var.name + '/gradients', grad)
            self.train_op = self.optimizer.apply_gradients(gradients, name="train_op")

        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update, name="target_network_update")
