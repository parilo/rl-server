import numpy as np
import tempfile
import tensorflow as tf
import threading

class RLTrainLoop ():

    def __init__ (self, num_actions, observation_size):

        self.num_actions = num_actions
        self.observation_size = observation_size

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.get_default_graph ()
        self.sess = tf.Session(config=config)
        self.logger = tf.summary.FileWriter("logs")

        batch_size = 128
        self.max_experience_size = 1000000
        self.start_learning_after = 500 # 20000

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

        self.exp_enqueue_op = all_experience.enqueue_many([
            self.inp_rewards,
            self.inp_actions,
            self.inp_prev_states,
            self.inp_next_states
        ])

        self.exp_size_op = all_experience.size ()

        [rewards, actions, prev_states, next_states] = all_experience.dequeue_many (batch_size)
        self.dequeued_rewards = rewards
        self.dequeued_actions = actions
        self.dequeued_prev_states = prev_states
        self.dequeued_next_states = next_states

        self.sum_rewards = 0

        self.train_ops = []

    def init_vars (self):

        self.logger.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def store_exp_batch (self, rewards, actions, prev_states, next_states):
        [_] = self.sess.run ([
            self.exp_enqueue_op
        ], {
            self.inp_rewards: np.array(rewards),
            self.inp_actions: np.array(actions),
            self.inp_prev_states: np.array(prev_states),
            self.inp_next_states: np.array(next_states)
        })
        self.sum_rewards += np.sum(rewards)

    def add_train_ops (self, train_ops_list):
        self.train_ops += train_ops_list

    def set_loss_op (self, loss_op):
        self.loss_op = loss_op

    def set_train_listener (self, listener):
        self.train_listener = listener

    def stop_train (self):
        self.coord.request_stop()

    def train (self):

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        def TrainLoop(coord):
            try:
                i = 0
                while not self.coord.should_stop():
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

                    i += 1
                    if i % 50 == 49:
                        # print ('trains: {} rewards: {} loss: {} stored: {}'.format(i, self.sum_rewards, loss, queue_size))
                        self.sum_rewards = 0

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                self.coord.request_stop()

        self.train_thread = threading.Thread(target=TrainLoop, args=(self.coord,))
        self.train_thread.start ()

    def join ():
        self.coord.join(threads)
        self.coord.join(train_thread)
        self.sess.close()
