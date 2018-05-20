import numpy as np
import tensorflow as tf
import threading
import time

class ExperienceReplayBuffer ():

    def __init__ (
        self,
        tf_session,
        observation_shapes,
        action_size,
        batch_size,
        experience_replay_buffer_size,
        experience_replay_buffer_size_min,
        store_every_nth,
        action_dtype = tf.float32,
        is_actions_space_continuous = True
    ):
        self.sess = tf_session
        self.num_actions = action_size
        self.observation_shapes = observation_shapes
        self.observation_batch_shapes = []
        self.num_of_observation_inputs = len(self.observation_shapes)
        for s in self.observation_shapes:
            self.observation_batch_shapes.append(
                tuple([None] + list(s))
            )

        self.batch_size = batch_size
        self.max_experience_size = experience_replay_buffer_size
        self.experience_replay_buffer_size_min = experience_replay_buffer_size_min
        self.store_every_nth = store_every_nth

        self.inp_rewards = tf.placeholder (tf.float32, (None,), name='inp_rewards')
        self.inp_actions = tf.placeholder (
            action_dtype, (None, self.num_actions) if is_actions_space_continuous else (None,),
            name='inp_actions'
        )
        self.inps_observations_for_act = []
        self.inps_prev_states = []
        self.inps_next_states = []
        for shape in self.observation_batch_shapes:
            self.inps_observations_for_act.append(tf.placeholder (tf.float32, shape, name='inps_observations_for_act'))
            self.inps_prev_states.append(tf.placeholder (tf.float32, shape, name='inp_prev_state'))
            self.inps_next_states.append(tf.placeholder (tf.float32, shape, name='inp_next_states'))
        self.inp_not_terminator = tf.placeholder (tf.float32, (None,), name='inp_not_terminator')

        all_experience = tf.RandomShuffleQueue (
            capacity = self.max_experience_size,
            min_after_dequeue = self.experience_replay_buffer_size_min,
            dtypes = [tf.float32, action_dtype] + [tf.float32, tf.float32] * self.num_of_observation_inputs + [tf.float32],
            shapes = [
                    (),
                    (self.num_actions,) if is_actions_space_continuous else (),
                ] +
                self.observation_shapes * 2 +
                [()]  # not terminator
        )

        self.exp_enqueue_op = all_experience.enqueue_many(
            [
                self.inp_rewards,
                self.inp_actions
            ] +
            self.inps_prev_states +
            self.inps_next_states +
            [self.inp_not_terminator]
        )

        self.exp_size_op = all_experience.size ()

        dequeued = all_experience.dequeue_many (self.batch_size, name='dequeue_from_experience_replay_buffer')
        self.dequeued_rewards = dequeued[0]
        self.dequeued_actions = dequeued[1]
        self.dequeued_prev_states = dequeued[2:2 + self.num_of_observation_inputs]
        self.dequeued_next_states = dequeued[2 + self.num_of_observation_inputs: 2 + 2*self.num_of_observation_inputs]
        self.dequeued_not_terminator = dequeued[-1]

        self.sum_rewards = 0
        self.store_invoked_count = store_every_nth
        self.stored_count = 0
        self.buffer_size = 0

    def get_stored_count(self):
        return self.stored_count

    def get_observations_for_act_placeholders(self):
        return self.inps_observations_for_act

    def get_train_batches(self):
        return (
            self.dequeued_rewards,
            self.dequeued_actions,
            self.dequeued_prev_states,
            self.dequeued_next_states,
            self.dequeued_not_terminator
        )

    def reset_sum_rewards(self):
        self.sum_rewards = 0

    def get_sum_rewards(self):
        return self.sum_rewards

        # self.train_ops = []
        # self.store_ops = []

        # self.train_listener = None
        # self.store_listener = None

    # """
    #     Split
    #     [
    #         [observation_part_1_agent_1, observation_part_2_agent_1, ...],
    #         [observation_part_1_agent_2, observation_part_2_agent_2, ...],
    #         ...
    #     ] into
    #     [
    #         [observation_part_1_agent_1, observation_part_1_agent_2, ...],
    #         [observation_part_2_agent_1, observation_part_2_agent_2, ...],
    #         ...
    #     ]
    #     function is used in putting into experience replay
    # """
    # def split_states_parts(self, states):
    #     states_parts = []
    #     for i in range(self.num_of_observation_inputs):
    #         states_parts.append([])
    #
    #     for s in states:
    #         for i in range(self.num_of_observation_inputs):
    #             states_parts[i].append(s[i])
    #     return states_parts

    def store_exp_batch (self, rewards, actions, prev_states, next_states, is_terminators):

        if self.store_invoked_count % self.store_every_nth == 0:

            is_terminators = np.array(is_terminators)
            is_not_terminators = np.ones_like(is_terminators) - is_terminators

            self.stored_count += len (rewards)
            # if self.stored_count < self.start_learning_after:
            #     print ('stored exp: {}'.format(self.stored_count))

            # prev_states_parts = self.split_states_parts(prev_states)
            # next_states_parts = self.split_states_parts(next_states)
            # print('--- prev_states_parts: {}'.format(np.array(prev_states_parts).shape))
            # print('--- next_states_parts: {}'.format(np.array(next_states_parts).shape))

            feed = {
                self.inp_rewards: np.array(rewards),
                self.inp_actions: np.array(actions),
                self.inp_not_terminator: is_not_terminators
            }
            for i in range (self.num_of_observation_inputs):
                feed[self.inps_prev_states[i]] = np.array(prev_states)
                feed[self.inps_next_states[i]] = np.array(next_states)

            [_, self.buffer_size] = self.sess.run (
                [self.exp_enqueue_op, self.exp_size_op],
                feed
            )
            self.sum_rewards += np.sum(rewards)

        self.store_invoked_count += 1

    """
        When you want to put back just dequeded batch
    """
    def put_back_exp_batch(self, rewards, actions, prev_states, next_states, is_not_terminators):
        feed = {
            self.inp_rewards: np.array(rewards),
            self.inp_actions: np.array(actions),
            self.inp_not_terminator: np.array(is_not_terminators)
        }
        for ii in range(self.num_of_observation_inputs):
            feed[self.inps_prev_states[ii]] = prev_states[ii]
            feed[self.inps_next_states[ii]] = next_states[ii]

        [_, self.buffer_size] = self.sess.run ([
            self.exp_enqueue_op,
            self.exp_size_op
        ], feed)
