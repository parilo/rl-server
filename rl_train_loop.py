import numpy as np
import tempfile
import tensorflow as tf
import threading
import time
from experience_replay_buffer import ExperienceReplayBuffer

class RLTrainLoop ():

    def __init__ (
        self,
        observation_shapes,
        action_size,
        action_dtype,
        is_actions_space_continuous,
        batch_size = 96,
        discount_rate = 0.99,
        experience_replay_buffer_size = 1000000,
        store_every_nth = 1,
        start_learning_after = 5000
    ):

        self.observation_shapes = observation_shapes
        self.action_size = action_size
        self.action_dtype = action_dtype
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.experience_replay_buffer_size = experience_replay_buffer_size
        self.start_learning_after = start_learning_after
        self.store_every_nth = store_every_nth

        self.num_of_observation_inputs = len(observation_shapes)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.logger = tf.summary.FileWriter("logs")

        self.exp_replay_buffer = ExperienceReplayBuffer(
            self.sess,
            self.observation_shapes,
            self.action_size,
            self.batch_size,
            self.experience_replay_buffer_size,
            self.start_learning_after,
            self.store_every_nth,
            action_dtype = self.action_dtype,
            is_actions_space_continuous = is_actions_space_continuous
        )

        self.train_ops = []
        self.store_ops = []

        self.train_listener = None
        self.store_listener = None

    def get_observations_for_act_placeholders(self):
        return self.exp_replay_buffer.get_observations_for_act_placeholders()

    def get_train_batches(self):
        return self.exp_replay_buffer.get_train_batches()

    def init_vars (self, model_load_callback=None):

        self.logger.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)
        if model_load_callback is not None:
            model_load_callback(self.sess, self.saver)

    def add_train_ops (self, train_ops_list):
        self.train_ops += train_ops_list

    def add_store_ops (self, store_ops_list):
        self.store_ops += store_ops_list

    def set_loss_op (self, loss_op):
        self.loss_op = loss_op

    def set_train_listener (self, listener):
        self.train_listener = listener

    def set_store_listener (self, listener):
        self.store_listener = listener

    def stop_train (self):
        self.coord.request_stop()

    def _start_train (self, loop_func):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        train_thread = threading.Thread(target=loop_func, args=(coord,))
        train_thread.start ()
        return coord, threads, train_thread

    def store_exp_batch (self, rewards, actions, prev_states, next_states, is_terminators):
        self.exp_replay_buffer.store_exp_batch(
            rewards, actions, prev_states, next_states, is_terminators
        )

    def train (self):

        def TrainLoop(coord):

            try:
                i = self.start_learning_after
                while not coord.should_stop():

                    stored_count = self.exp_replay_buffer.get_stored_count()
                    if i > stored_count:
                        print ('--- waiting for exp: {} {}'.format(i, stored_count))
                        time.sleep(1.0)
                        continue

                    (batch_rewards,
                    batch_actions,
                    batch_prev_states,
                    batch_next_states,
                    batch_not_terminator) = self.get_train_batches()

                    self.train_outputs = self.sess.run(
                        [
                            batch_rewards,
                            batch_actions,
                        ] +
                        batch_prev_states +
                        batch_next_states +
                        [
                            batch_not_terminator,
                            self.exp_replay_buffer.exp_size_op,
                            self.loss_op
                        ] + self.train_ops
                    )

                    r = self.train_outputs [0]
                    a = self.train_outputs [1]
                    ps = self.train_outputs [2: 2 + self.num_of_observation_inputs]
                    ns = self.train_outputs [2 + self.num_of_observation_inputs: 2 + 2*self.num_of_observation_inputs]
                    nt = self.train_outputs [2 + 2*self.num_of_observation_inputs]
                    queue_size = self.train_outputs [2 + 2*self.num_of_observation_inputs + 1]
                    loss = self.train_outputs [2 + 2*self.num_of_observation_inputs + 2]

                    if self.train_listener is not None:
                        self.train_listener ()

                    if queue_size < self.experience_replay_buffer_size - 10000:

                        self.exp_replay_buffer.put_back_exp_batch(
                            r, a, ps, ns, nt
                        )

                    if i % 2000 == 1999:
                        print ('trains: {} rewards: {} loss: {} stored: {}'.format(i, self.exp_replay_buffer.get_sum_rewards(), loss, queue_size))
                        self.exp_replay_buffer.reset_sum_rewards()

                    if i % 10000 == 0:
                        save_path = self.saver.save(self.sess, 'ckpt/model-{}.ckpt'.format(i))
                        print("Model saved in file: %s" % save_path)

                    i += 1


            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

        self.coord, self.threads, self.train_thread = self._start_train (TrainLoop)

    def join (self):
        self.coord.join(self.threads)
        self.fifo_coord.join(self.fifo_threads)
        self.sess.close()
