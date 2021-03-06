import numpy as np
import tempfile
import tensorflow as tf
import threading
import time

class RLTrainLoop ():

    def __init__ (self, num_actions, observation_size):

        self.num_actions = num_actions
        self.observation_size = observation_size

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.get_default_graph ()
        self.sess = tf.Session(config=config)
        self.logger = tf.summary.FileWriter("logs")

        self.batch_size = 64
        self.max_experience_size = 1000000
        self.start_learning_after = 5000
        self.store_every_nth = 5

        self.inp_rewards = tf.placeholder (tf.float32, (None,))
        self.inp_actions = tf.placeholder (tf.float32, (None, num_actions))
        self.inp_prev_states = tf.placeholder (tf.float32, (None, observation_size))
        self.inp_next_states = tf.placeholder (tf.float32, (None, observation_size))

        all_experience = tf.RandomShuffleQueue (
            capacity=self.max_experience_size,
            min_after_dequeue=self.start_learning_after,
            dtypes=[tf.float32, tf.float32, tf.float32, tf.float32],
            shapes=[
                (),
                (num_actions,),
                (observation_size,),
                (observation_size,)
            ]
        )

        exp_fifo_queue = tf.FIFOQueue (
            capacity=self.max_experience_size,
            dtypes=[tf.float32, tf.float32, tf.float32, tf.float32],
            shapes=[
                (),
                (num_actions,),
                (observation_size,),
                (observation_size,)
            ]
        )

        self.exp_enqueue_op = all_experience.enqueue_many([
            self.inp_rewards,
            self.inp_actions,
            self.inp_prev_states,
            self.inp_next_states
        ])

        self.exp_fifo_enqueue_op = exp_fifo_queue.enqueue_many([
            self.inp_rewards,
            self.inp_actions,
            self.inp_prev_states,
            self.inp_next_states
        ])

        self.exp_size_op = all_experience.size ()

        [rewards, actions, prev_states, next_states] = all_experience.dequeue_many (self.batch_size)
        self.dequeued_rewards = rewards
        self.dequeued_actions = actions
        self.dequeued_prev_states = prev_states
        self.dequeued_next_states = next_states

        [fifo_rewards, fifo_actions, fifo_prev_states, fifo_next_states] = exp_fifo_queue.dequeue_many (self.batch_size)
        self.dequeued_fifo_rewards = fifo_rewards
        self.dequeued_fifo_actions = fifo_actions
        self.dequeued_fifo_prev_states = fifo_prev_states
        self.dequeued_fifo_next_states = fifo_next_states
        self.exp_fifo_size_op = exp_fifo_queue.size ()

        self.sum_rewards = 0
        self.store_index = 0
        self.stored_count = 0

        self.train_ops = []
        self.store_ops = []

        self.train_listener = None
        self.store_listener = None

    def init_vars (self):

        self.logger.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

    def store_exp_batch (self, rewards, actions, prev_states, next_states):
        self.stored_count += len (rewards)
        if self.stored_count < self.start_learning_after:
            print ('stored exp: {}'.format(self.stored_count))

        self.store_outputs = self.sess.run ([
            self.exp_enqueue_op,
            self.inp_rewards,
            self.inp_actions,
            self.inp_prev_states,
            self.inp_next_states
        ] + self.store_ops, {
            self.inp_rewards: np.array(rewards),
            self.inp_actions: np.array(actions),
            self.inp_prev_states: np.array(prev_states),
            self.inp_next_states: np.array(next_states)
        })
        self.sum_rewards += np.sum(rewards)
        if self.store_listener is not None:
            self.store_listener ()

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

        for v in tf.trainable_variables():
            print ("--- all v: " + v.name)

        def TrainLoop(coord):

            try:
                i = self.start_learning_after
                while not coord.should_stop():

                    if i > self.stored_count:
                        print ('--- waiting for exp: {} {}'.format(i, self.stored_count))
                        time.sleep(1.0)
                        continue

                    self.train_outputs = self.sess.run([
                        self.dequeued_rewards,
                        self.dequeued_actions,
                        self.dequeued_prev_states,
                        self.dequeued_next_states,
                        self.exp_size_op,
                        self.loss_op
                    ] + self.train_ops)

                    r = self.train_outputs [0]
                    a = self.train_outputs [1]
                    ps = self.train_outputs [2]
                    ns = self.train_outputs [3]
                    queue_size = self.train_outputs [4]
                    loss = self.train_outputs [5]

                    if self.train_listener is not None:
                        self.train_listener ()

                    if queue_size < self.max_experience_size - 10000:
                        [_, size] = self.sess.run ([
                            self.exp_enqueue_op,
                            self.exp_size_op
                        ], {
                            self.inp_rewards: r,
                            self.inp_actions: a,
                            self.inp_prev_states: ps,
                            self.inp_next_states: ns
                        })

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
