import numpy as np
import tempfile
import tensorflow as tf
import threading
import time

class RLTrainLoop ():

    def __init__ (self, num_actions, observation_shapes):

        self.num_actions = num_actions
        self.observation_shapes = observation_shapes
        self.observation_batch_shapes = []
        self.num_of_observation_inputs = len(self.observation_shapes)
        for s in self.observation_shapes:
            self.observation_batch_shapes.append(
                tuple([None] + list(s))
            )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.get_default_graph ()
        self.sess = tf.Session(config=config)
        self.logger = tf.summary.FileWriter("logs")

        self.batch_size = 96
        self.max_experience_size = 3000000
        self.start_learning_after = 5000
        self.store_every_nth = 1

        self.inp_rewards = tf.placeholder (tf.float32, (None,), name='inp_rewards')
        self.inp_actions = tf.placeholder (tf.float32, (None, num_actions), name='inp_actions')
        self.inps_prev_states = []
        self.inps_next_states = []
        for shape in self.observation_batch_shapes:
            self.inps_prev_states.append(tf.placeholder (tf.float32, shape, name='inp_prev_state'))
            self.inps_next_states.append(tf.placeholder (tf.float32, shape, name='inp_next_states'))
        self.inp_not_terminator = tf.placeholder (tf.float32, (None,), name='inp_not_terminator')

        all_experience = tf.RandomShuffleQueue (
            capacity=self.max_experience_size,
            min_after_dequeue=self.start_learning_after,
            dtypes=[tf.float32, tf.float32] + [tf.float32, tf.float32] * self.num_of_observation_inputs + [tf.float32],
            shapes=[
                    (),
                    (num_actions,),
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

        dequeued = all_experience.dequeue_many (self.batch_size)
        self.dequeued_rewards = dequeued[0]
        self.dequeued_actions = dequeued[1]
        self.dequeued_prev_states = dequeued[2:2 + self.num_of_observation_inputs]
        self.dequeued_next_states = dequeued[2 + self.num_of_observation_inputs: 2 + 2*self.num_of_observation_inputs]
        self.dequeued_not_terminator = dequeued[-1]

        self.sum_rewards = 0
        self.store_index = 0
        self.stored_count = 0

        self.train_ops = []
        self.store_ops = []

        self.train_listener = None
        self.store_listener = None

    def init_vars (self, model_load_callback=None):

        self.logger.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)
        if model_load_callback is not None:
            model_load_callback(self.sess, self.saver)

    def split_states_parts(self, states):
        states_parts = []
        for i in range(self.num_of_observation_inputs):
            states_parts.append([])

        for s in states:
            for i in range(self.num_of_observation_inputs):
                states_parts[i].append(s[i])
        return states_parts

    def store_exp_batch (self, rewards, actions, prev_states, next_states, is_terminators):

        if self.store_index % self.store_every_nth == 0:

            is_terminators = np.array(is_terminators)
            is_not_terminators = np.ones_like(is_terminators) - is_terminators

            self.stored_count += len (rewards)
            if self.stored_count < self.start_learning_after:
                print ('stored exp: {}'.format(self.stored_count))

            # prev_states_parts = self.split_states_parts(prev_states)
            # next_states_parts = self.split_states_parts(next_states)
            # print('--- prev_states_parts: {}'.format(np.array(prev_states_parts).shape))
            # print('--- next_states_parts: {}'.format(np.array(next_states_parts).shape))
            feed = {
                self.inp_rewards: np.array(rewards),
                self.inp_actions: np.array(actions)
            }
            for i in range(self.num_of_observation_inputs):
                feed[self.inps_prev_states[i]] = np.array(prev_states)
                feed[self.inps_next_states[i]] = np.array(next_states)

            feed[self.inp_not_terminator] = is_not_terminators

            self.store_outputs = self.sess.run (
                [
                    self.exp_enqueue_op,
                    self.inp_rewards,
                    self.inp_actions
                ] +
                self.inps_prev_states +
                self.inps_next_states +
                [self.inp_not_terminator] +
                self.store_ops,
                feed
            )
            self.sum_rewards += np.sum(rewards)
            if self.store_listener is not None:
                self.store_listener ()

        self.store_index += 1

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

    def train (self):

        # for v in tf.trainable_variables():
        #     print ("--- all v: " + v.name)

        def TrainLoop(coord):

            try:
                i = self.start_learning_after
                while not coord.should_stop():

                    if i > self.stored_count:
                        print ('--- waiting for exp: {} {}'.format(i, self.stored_count))
                        time.sleep(1.0)
                        continue

                    self.train_outputs = self.sess.run(
                        [
                            self.dequeued_rewards,
                            self.dequeued_actions,
                        ] +
                        self.dequeued_prev_states +
                        self.dequeued_next_states +
                        [
                            self.dequeued_not_terminator,
                            self.exp_size_op,
                            self.loss_op
                        ] + self.train_ops
                        # , {
                        #     self.is_learning_phase: 1
                        # }
                    )

                    r = self.train_outputs [0]
                    a = self.train_outputs [1]
                    ps = self.train_outputs [2: 2 + self.num_of_observation_inputs]
                    ns = self.train_outputs [2 + self.num_of_observation_inputs: 2 + 2*self.num_of_observation_inputs]
                    nt = self.train_outputs [2 + 2*self.num_of_observation_inputs]
                    queue_size = self.train_outputs [2 + 2*self.num_of_observation_inputs + 1]
                    loss = self.train_outputs [2 + 2*self.num_of_observation_inputs + 2]


                    # self.train_outputs = self.sess.run([
                    #     self.dequeued_rewards,
                    #     self.dequeued_actions,
                    #     self.dequeued_prev_states,
                    #     self.dequeued_next_states,
                    #     self.dequeued_not_terminator,
                    #     self.exp_size_op,
                    #     self.loss_op
                    # ] + self.train_ops)
                    #
                    # r = self.train_outputs [0]
                    # a = self.train_outputs [1]
                    # ps = self.train_outputs [2]
                    # ns = self.train_outputs [3]
                    # nt = self.train_outputs [4]
                    # queue_size = self.train_outputs [5]
                    # loss = self.train_outputs [6]

                    if self.train_listener is not None:
                        self.train_listener ()

                    if queue_size < self.max_experience_size - 10000:

                        feed = {
                            self.inp_rewards: np.array(r),
                            self.inp_actions: np.array(a),
                            self.inp_not_terminator: np.array(nt)
                        }
                        for ii in range(self.num_of_observation_inputs):
                            feed[self.inps_prev_states[ii]] = ps[ii]
                            feed[self.inps_next_states[ii]] = ns[ii]

                        [_, size] = self.sess.run ([
                            self.exp_enqueue_op,
                            self.exp_size_op
                        ], feed)

                    # if queue_size < self.max_experience_size - 10000:
                    #     [_, size] = self.sess.run ([
                    #         self.exp_enqueue_op,
                    #         self.exp_size_op
                    #     ], {
                    #         self.inp_rewards: r,
                    #         self.inp_actions: a,
                    #         self.inp_prev_states: ps,
                    #         self.inp_next_states: ns,
                    #         self.inp_not_terminator: nt
                    #     })

                    if i % 2000 == 1999:
                        print ('trains: {} rewards: {} loss: {} stored: {}'.format(i, self.sum_rewards, loss, queue_size))
                        self.sum_rewards = 0

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
