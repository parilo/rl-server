import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import ContinuousDeepQ
from tf_rl import simulate
from tf_rl.models import MLP

class Quadrotor2D ():

    def __init__ (self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        journalist = tf.summary.FileWriter("logs")

        observation_size = 50;
        observations_in_seq = 1;
        input_size = observation_size*observations_in_seq;
        num_actions = 2;

        batch_size = 128
        self.max_experience_size = 3000000
        self.start_learning_after = 20000
        learning_rate=1e-4

        r = tf.nn.relu
        t = tf.nn.tanh

        critic = MLP([input_size, num_actions], [2048, 512, 256, 256, 1],
                    [t, t, t, t, tf.identity], scope='critic')

        self.actor = MLP([input_size,], [2048, 512, 256, 256, num_actions],
                    [t, t, t, t, tf.identity], scope='actor')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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
        self.q_rewards = rewards
        self.q_actions = actions
        self.q_prev_states = prev_states
        self.q_next_states = next_states

        self.controller = ContinuousDeepQ(
            input_size,
            num_actions,
            self.actor,
            critic,
            optimizer,
            self.sess,
            discount_rate=0.99,
            target_actor_update_rate=0.01,
            target_critic_update_rate=0.01,
            exploration_period=5000,
            max_experience=10000,
            store_every_nth=4,
            train_every_nth=4,
            summary_writer=journalist,
            rewards = self.q_rewards,
            given_action = self.q_actions,
            observation = self.q_prev_states,
            next_observation = self.q_next_states,
            next_observation_mask = tf.ones((batch_size,), tf.float32)
        )

        self.sess.run(tf.global_variables_initializer())

        journalist.add_graph(self.sess.graph)

        self.sum_rewards = 0

    def act (self, state):
        [a] = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(state).reshape((1,-1))
            }
        )
        return a.tolist ()

    def act_batch (self, states):
        a = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(states)
            }
        )
        return a.tolist ()

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

    def train (self):

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        def TrainLoop(coord):
            try:
                i = 0
                while not coord.should_stop():
                    r, a, ps, ns, loss, _, _, _, queue_size = self.sess.run([
                        self.q_rewards,
                        self.q_actions,
                        self.q_prev_states,
                        self.q_next_states,
                        self.controller.critic_error,
                        self.controller.actor_update,
                        self.controller.critic_update,
                        self.controller.update_all_targets,
                        self.exp_size_op
                    ])

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
                        print ('trains: {} rewards: {} loss: {} stored: {}'.format(i, self.sum_rewards, loss, queue_size))
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
