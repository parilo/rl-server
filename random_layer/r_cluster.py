import tensorflow as tf
import numpy as np
import cv2

class RCluster ():

    def __init__ (
        self,
        train_loop,
        map_side_size,
        initial_limits,
        input_dim,
        samples_tensor,
        scope
    ):
        self.train_loop = train_loop
        self.map_side_size = map_side_size
        self.initial_limits = initial_limits
        self.clusters_count = map_side_size * map_side_size
        self.input_dim = input_dim
        self.scope = scope
        self.highlighted_vector = np.zeros((1, self.input_dim))
        self.centroids = None

        # self.som = SOM(
        #     (self.input_dim,),
        #     self.map_side_size,
        #     1,
        #     train_loop.sess,
        #     samples_tensor
        # )

        with tf.variable_scope(self.scope):
            n_initializer =  tf.random_uniform_initializer(self.initial_limits[0], self.initial_limits[1])
            self.neurons = tf.get_variable('neurons', (self.clusters_count, self.input_dim), initializer=n_initializer)
            # self.normalize_neurons = tf.assign(self.neurons, tf.div (self.neurons, tf.norm(self.neurons, ord='euclidean', axis=1)))
            # self.normalize_neurons = tf.assign(self.neurons,
            #     tf.nn.l2_normalize(self.neurons, dim=1, epsilon=1e-12) * (self.initial_limits[1] - self.initial_limits[0]) + self.initial_limits[0]
            # )
            #
            # self.rescale_neurons = tf.assign(self.neurons,
            #     self.neurons * (self.initial_limits[1] - self.initial_limits[0]) + self.initial_limits[0]
            # )

            self.input_batch = tf.placeholder(tf.float32, (None, self.input_dim))
            batch_size = tf.shape(self.input_batch)[0]
            input_tiled = tf.reshape(tf.tile(self.input_batch, [1, self.clusters_count]), [batch_size, self.clusters_count, self.input_dim])
            neurons_tiled = tf.reshape(tf.tile(self.neurons, [batch_size, 1]), [batch_size, self.clusters_count, self.input_dim])
            diff = tf.subtract(input_tiled, neurons_tiled)
            self.batch_winners = tf.argmin(tf.reduce_sum(tf.multiply(diff, diff), axis=2), axis=1)
            # self.batch_winners = tf.argmax(tf.tensordot(self.inp ut_batch, self.neurons, [[1], [1]]), 1)

    def init(self):
        print(self.train_loop.sess.run(self.neurons))

    # def normalize(self):
    #     print('--- normalizing neurons')
    #     self.train_loop.sess.run(self.normalize_neurons)
    #     print(self.train_loop.sess.run(self.neurons))
    #
    # def rescale(self):
    #     print('--- rescale neurons')
    #     self.train_loop.sess.run(self.rescale_neurons)
    #     print(self.train_loop.sess.run(self.neurons))

    def _set_centroids (self, centroids):
        self.centroids = centroids

    def get_centroids (self):
        return self.centroids

    def get_clusters_count (self):
        return self.clusters_count

    def get_train_ops (self):
        return []

    def get_clusters (self, vectors):
        return self.train_loop.sess.run(self.batch_winners, {self.input_batch: vectors})

    def get_neurons (self):
        return self.train_loop.sess.run(self.neurons)

    def show_centroids (self, image_grid):
        # cmax = np.max(image_grid)
        # cmin = np.min(image_grid)
        # image_grid = (image_grid - cmin) / (cmax - cmin)
        #
        # index = self.som.get_batch_winner (self.highlighted_vector) [0]
        # highlighted = divmod(index, self.map_side_size)
        # image_grid [highlighted[0], highlighted[1], :] = (1, 1, 1)
        #
        # # print (image_grid)
        cv2.imshow(self.scope, np.zeros((20, 20, 3)))
        cv2.waitKey (1)
        pass

    def highlight (self, vector):
        # self.highlighted_vector = vector.reshape((1, -1))
        pass
