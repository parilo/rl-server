
# author Anton Pechenko
# forpost78@gmail.com

import numpy as np
import random
import tensorflow as tf

from collections import deque

class CriticControl(object):
    def __init__(self, observation_shape,
                       action_shape,
                       batch_size,
                    #    actor,
                       critic,
                       optimizer,
                       discount_rate=0.95,
                    #    target_actor_update_rate=0.01,
                       target_critic_update_rate=0.01,
                       rewards = None,
                       given_action = None,
                       observation_for_act = None,
                       observation = None,
                       next_observation = None,
                       next_observation_mask = None
                       ):
        """Initialized the Deepq object.
        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        action_size : int
            length of the vector representing an action
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, action_size]
        optimizer: tf.solver.*
            optimizer for prediction error
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        target_actor_update_rate: float
            how much to update target critci after each
            iteration. Let's call target_critic_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        target_critic_update_rate: float
            analogous to target_actor_update_rate, but for
            target_critic
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        # self.observation_size          = observation_size
        # self.action_size               = action_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.batch_size = batch_size

        # self.actor                     = actor
        self.critic                    = critic
        self.optimizer                 = optimizer

        self.discount_rate             = tf.constant(discount_rate)

        # self.target_actor_update_rate = \
        #         tf.constant(target_actor_update_rate)
        self.target_critic_update_rate = \
                tf.constant(target_critic_update_rate)

        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards") if (rewards is None) else rewards
        self.given_action = tf.placeholder(tf.float32, action_shape, name="given_action") if given_action is None else given_action
        self.observation_for_act = tf.placeholder(tf.float32, observation_shape, name="observation_for_act") if observation_for_act is None else observation_for_act
        self.observation = tf.placeholder(tf.float32, observation_shape, name="observation") if observation is None else observation
        self.next_observation = tf.placeholder(tf.float32, observation_shape, name="next_observation") if next_observation is None else next_observation
        self.next_observation_mask = tf.placeholder(tf.float32, (None,), name="next_observation_mask") if next_observation_mask is None else next_observation_mask

        self.create_variables()

    @staticmethod
    def update_target_network(source_network, target_network, update_rate):
        target_network_update = []
        for v in source_network.variables():
            # this is equivalent to target = (1-alpha) * target + alpha * source
            # print ("source: " + v.name + " : " + str(v.get_shape()))
            pass
        for v in target_network.variables():
            # this is equivalent to target = (1-alpha) * target + alpha * source
            # print ("target: " + v.name + " : " + str(v.get_shape()))
            pass
        for v_source, v_target in zip(source_network.variables(), target_network.variables()):
            # this is equivalent to target = (1-alpha) * target + alpha * source
            update_op = v_target.assign_sub(update_rate * (v_target - v_source))
            target_network_update.append(update_op)
        return tf.group(*target_network_update)

    def _get_action_values(self, critic, observation, batch_size, proposals_count = 100):
        num_actions = self.action_shape[0]
        observation_size = self.observation_shape[1]
        print('--- observation: {}'.format(observation))
        actions_proposals_shape = [batch_size, proposals_count, num_actions]
        actions_proposals = tf.random_uniform(actions_proposals_shape, minval=-1.0, maxval=1.0)
        print('--- actions_proposals: {}'.format(actions_proposals))
        observations_for_proposals = tf.tile(tf.expand_dims(observation, 1), [1, proposals_count, 1])
        print('--- observations_for_proposals: {}'.format(observations_for_proposals))
        actions_values = tf.reshape(
            critic([
                tf.reshape(observations_for_proposals, [-1, observation_size]),
                tf.reshape(actions_proposals, [-1, num_actions])]),
            [batch_size, proposals_count, 1]
        )
        print('--- actions_values: {}'.format(actions_values))
        return actions_values, actions_proposals

    def act(self, critic, observation, batch_size, proposals_count = 100):
        actions_values, actions_proposals = self._get_action_values(critic, observation, batch_size, proposals_count)
        max_value_index = tf.argmax(actions_values, axis=1, output_type=tf.int32)
        print('--- max_value_index: {}'.format(max_value_index))

        indices = tf.concat(
            [
                tf.reshape(tf.range(0, batch_size), [batch_size, 1]),
                max_value_index
            ],
            axis=1
        )
        print('--- indices: {}'.format(indices))

        selected_action = tf.gather_nd(actions_proposals, indices)
        print('--- selected_action: {}'.format(selected_action))
        return selected_action

    def act_boltzmann_exploration(self, critic, observation, batch_size, proposals_count = 100):

        act_count = tf.Variable(0, trainable=False, dtype=tf.int64)
        act_count_increase = tf.assign(act_count, act_count + 1)
        self.t = 1.0 - (1.0 - 0.01) / 500000.0 * tf.cast(act_count, tf.float32)

        actions_values, actions_proposals = self._get_action_values(critic, observation, batch_size, proposals_count)
        actions_values_logits = tf.reshape(actions_values, [1, -1]) / self.t
        print('--- actions_values_logits {}'.format(actions_values_logits))
        sampled_action = tf.cast(tf.multinomial(logits=actions_values_logits, num_samples=1), tf.int32)
        print('--- samples action {}'.format(sampled_action))

        indices = tf.concat(
            [
                tf.reshape(tf.range(0, batch_size), [batch_size, 1]),
                sampled_action
            ],
            axis=1
        )
        print('--- indices: {}'.format(indices))

        selected_action = tf.gather_nd(actions_proposals, indices)
        print('--- selected_action: {}'.format(selected_action))
        return selected_action, act_count_increase


    # def mc_sum_over_actions(self, critic, observation, batch_size, proposals_count = 100):
    #     num_actions = self.action_shape[0]
    #     observation_size = self.observation_shape[1]
    #     print('--- observation: {}'.format(observation))
    #     actions_proposals_shape = [batch_size, proposals_count, num_actions]
    #     actions_proposals = tf.random_uniform(actions_proposals_shape, minval=-1.0, maxval=1.0)
    #     print('--- actions_proposals: {}'.format(actions_proposals))
    #     observations_for_proposals = tf.tile(tf.expand_dims(observation, 1), [1, proposals_count, 1])
    #     print('--- observations_for_proposals: {}'.format(observations_for_proposals))
    #     actions_values = tf.reshape(
    #         critic([
    #             tf.reshape(observations_for_proposals, [-1, observation_size]),
    #             tf.reshape(actions_proposals, [-1, num_actions])]),
    #         [batch_size, proposals_count, 1]
    #     )
    #     print('--- actions_values: {}'.format(actions_values))
    #     sum_over_actions = tf.reduce_mean(actions_values, axis=1)
    #     print('--- sum_over_actions: {}'.format(sum_over_actions))
    #     return sum_over_actions

    def mc_max_over_actions(self, critic, observation, batch_size, proposals_count = 100):
        num_actions = self.action_shape[0]
        observation_size = self.observation_shape[1]
        print('--- observation: {}'.format(observation))
        actions_proposals_shape = [batch_size, proposals_count, num_actions]
        actions_proposals = tf.random_uniform(actions_proposals_shape, minval=-1.0, maxval=1.0)
        print('--- actions_proposals: {}'.format(actions_proposals))
        observations_for_proposals = tf.tile(tf.expand_dims(observation, 1), [1, proposals_count, 1])
        print('--- observations_for_proposals: {}'.format(observations_for_proposals))
        actions_values = tf.reshape(
            critic([
                tf.reshape(observations_for_proposals, [-1, observation_size]),
                tf.reshape(actions_proposals, [-1, num_actions])]),
            [batch_size, proposals_count, 1]
        )
        print('--- actions_values: {}'.format(actions_values))
        max_value_index = tf.argmax(actions_values, axis=1, output_type=tf.int32)
        print('--- max_value_index: {}'.format(max_value_index))

        indices = tf.concat(
            [
                tf.reshape(tf.range(0, batch_size), [batch_size, 1]),
                max_value_index
            ],
            axis=1
        )
        print('--- indices: {}'.format(indices))

        max_action_values = tf.gather_nd(actions_values, indices)
        print('--- max_action_values: {}'.format(max_action_values))
        return max_action_values

    def create_variables(self):
        # self.target_actor  = self.actor.copy(scope="target_actor")
        self.target_critic = self.critic.copy(scope="target_critic")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.actor_val, self.act_count_increase = self.act_boltzmann_exploration(self.critic, self.observation_for_act, 1)
            print('                           ')
            # self.act_boltzmann_exploration(self.critic, self.observation_for_act, 1)
            # print('                           ')

#             # self.observation  = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
# #            self.actor_val = tf.nn.sigmoid(self.actor(self.observation)) * 40 - 20;
#             self.actor_val = self.actor(self.observation_for_act)
#             self.actor_action = tf.identity(self.actor_val, name="actor_action")
# #            tf.histogram_summary("actions", self.actor_action)

        # FOR PREDICTING TARGET FUTURE REWARDS
        with tf.name_scope("estimating_future_reward"):
            # self.next_observation          = tf.placeholder(tf.float32, (None, self.observation_size), name="next_observation")
            # self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            # self.next_action               = self.target_actor(self.next_observation) # ST
            # self.next_action               = tf.stop_gradient(self.target_actor(self.next_observation)) # ST

            # self.next_action = tf.stop_gradient(self.act(self.target_critic, self.next_observation, self.batch_size))
            # print('--- next_action: {}'.format(self.next_action))

#            print "next action: " + str(self.next_action)
            # tf.histogram_summary("target_actions", self.next_action)
            # self.next_value                = self.target_critic([self.next_observation, self.next_action]) # ST

            # self.next_value                = tf.stop_gradient(
            #     tf.reshape(
            #         self.target_critic([self.next_observation, self.next_action]),
            #         [-1]
            #     )
            # ) # ST

            self.next_value                = tf.stop_gradient(
                tf.reshape(
                    self.mc_max_over_actions(self.target_critic, self.next_observation, self.batch_size),
                    [-1]
                )
            ) # ST
            print('--- next_value: {}'.format(self.next_value))

            # self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            self.future_reward             = self.rewards + self.discount_rate *  self.next_observation_mask * self.next_value

        with tf.name_scope("q_for_act_calc"):

            self.value_given_action_for_act = self.critic([self.observation_for_act, self.actor_val])
            print('--- value_given_action_for_act: {}'.format(self.value_given_action_for_act))

        # with tf.name_scope("td_error_calc"):
        #
        #     self.next_value_for_act                = tf.stop_gradient(
        #         tf.reshape(
        #             self.mc_max_over_actions(self.target_critic, self.observation_for_act, 1),
        #             [-1]
        #         )
        #     ) # ST
        #     print('--- next_value_for_act: {}'.format(self.next_value_for_act))
        #
        #     self.future_reward             = self.rewards + self.discount_rate *  self.next_observation_mask * self.next_value
        #
        #     self.value_given_action_for_act         = tf.reshape(
        #         self.critic([self.observation_for_act, self.actor_val]),
        #         [-1]
        #     )
        #
        #     self.td_error_for_act = self.future_reward - self.value_given_action_for_act

        with tf.name_scope("critic_update"):
            ##### ERROR FUNCTION #####
            # self.given_action               = tf.placeholder(tf.float32, (None, self.action_size), name="given_action")
            self.value_given_action         = tf.reshape(
                self.critic([self.observation, self.given_action]),
                [-1]
            )
            # tf.scalar_summary("value_for_given_action", tf.reduce_mean(self.value_given_action))
            # temp_diff                       = self.value_given_action - self.future_reward
            # self.critic_error               = tf.identity(tf.reduce_mean(tf.square(temp_diff)), name='critic_error')

            self.critic_error               = tf.identity(tf.losses.huber_loss(self.value_given_action, self.future_reward), name='critic_error')
            ##### OPTIMIZATION #####
            critic_gradients                       = self.optimizer.compute_gradients(self.critic_error, var_list=self.critic.variables())
            # Add histograms for gradients.
            for grad, var in critic_gradients:
                # tf.histogram_summary('critic_update/' + var.name, var)
                if grad is not None:
                    # tf.histogram_summary('critic_update/' + var.name + '/gradients', grad)
                    pass
            self.critic_update              = self.optimizer.apply_gradients(critic_gradients, name='critic_train_op')
            # tf.scalar_summary("critic_error", self.critic_error)

        # with tf.name_scope("actor_update"):
        #     ##### ERROR FUNCTION #####
        #     # self.actor_score = self.critic([self.observation, self.actor_action])
        #     self.actor_score = self.critic([self.observation, self.actor(self.observation)])
        #
        #     ##### OPTIMIZATION #####
        #     # here we are maximizing actor score.
        #     # only optimize actor variables here, while keeping critic constant
        #     actor_gradients = self.optimizer.compute_gradients(tf.reduce_mean(-self.actor_score), var_list=self.actor.variables())
        #     # Add histograms for gradients.
        #     for grad, var in actor_gradients:
        #         # tf.histogram_summary('actor_update/' + var.name, var)
        #         if grad is not None:
        #             # tf.histogram_summary('actor_update/' + var.name + '/gradients', grad)
        #             pass
        #     self.actor_update              = self.optimizer.apply_gradients(actor_gradients, name='actor_train_op')
        #     # tf.scalar_summary("actor_score", tf.reduce_mean(self.actor_score))

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            # self.target_actor_update  = ContinuousDeepQ.update_target_network(self.actor, self.target_actor, self.target_actor_update_rate)
            self.target_critic_update = CriticControl.update_target_network(self.critic, self.target_critic, self.target_critic_update_rate)
            # self.update_all_targets = tf.group(self.target_actor_update, self.target_critic_update, name='target_networks_update')
            # self.update_all_targets = self.target_critic_update
