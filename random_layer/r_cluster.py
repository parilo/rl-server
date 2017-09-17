import tensorflow as tf
import numpy as np
import cv2

class RCluster ():

    def __init__ (
        self,
        train_loop,
        map_side_size,
        # initial_limits,
        initial_mean,
        initial_std,
        input_dim,
        samples_tensor,
        scope
    ):
        self.train_loop = train_loop
        self.map_side_size = map_side_size
        # self.initial_limits = initial_limits
        self.initial_mean = initial_mean
        self.initial_std = initial_std
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
            # n_initializer =  tf.random_uniform_initializer(self.initial_limits[0], self.initial_limits[1])
            n_initializer =  tf.random_uniform_initializer(0, 1)
            self.neurons = tf.get_variable('neurons', (self.clusters_count, self.input_dim), initializer=n_initializer)

            init_limit_a = self.initial_mean - self.initial_std
            init_limit_a = np.repeat(init_limit_a.reshape(1, self.input_dim), self.clusters_count, axis=0)
            init_limit_b = self.initial_mean + self.initial_std
            init_limit_b = np.repeat(init_limit_b.reshape(1, self.input_dim), self.clusters_count, axis=0)

            self.init_neurons = tf.assign(self.neurons, tf.multiply(self.neurons, init_limit_b - init_limit_a) + init_limit_a)

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

            self.replace_from_index = tf.placeholder(tf.int32, (50))
            self.replace_to_index = tf.placeholder(tf.int32, (50))
            gathered = tf.gather(self.neurons, self.replace_from_index)
            gathered = gathered + tf.random_uniform((50, self.input_dim), minval=-0.05, maxval=0.05) * np.repeat(self.initial_std.reshape(1, self.input_dim), 50, axis=0)
            self.update_neurons = tf.scatter_update(self.neurons, self.replace_to_index, gathered)

        self.updatable = False
        self.reset_stat()

    def set_updatable(self, updatable):
        self.updatable = updatable

    def init(self):
        self.train_loop.sess.run(self.init_neurons)
        # n = self.train_loop.sess.run(self.neurons)
        # print(n)
        # for i in range(self.clusters_count):
        #     c = self.get_clusters([n[i]])[0]
        #     print ('check item: {} c: {} {}'.format(i, c, n[i]))


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

    def update_stat (self, clusters):
        if self.updatable:
            for c in clusters:
                self.usage_stat [c] += 1
            self.usage_stat_iters += len(clusters)
            if self.usage_stat_iters > 100000:
                self.adapt_neurons()
                self.reset_stat()

    def reset_stat (self):
        self.usage_stat = np.zeros((self.clusters_count))
        self.usage_stat_iters = 0

    def adapt_neurons (self):
        max_usage = np.max(self.usage_stat)
        print('--- usage stat: min: {} max: {} non zero count: {}'.format(np.min(self.usage_stat), max_usage, np.count_nonzero(self.usage_stat)))
        if max_usage > 1000:
            # print(self.usage_stat.tolist())
            # stat = np.stack([range(self.clusters_count), self.usage_stat])
            # stat = np.sort(stat, axis=1)
            # print(np.argsort(self.usage_stat).tolist())
            # print(self.usage_stat.tolist())
            sorted_indices = np.argsort(self.usage_stat)
            less_used = sorted_indices[:50]
            more_used = sorted_indices[-50:]
            self.train_loop.sess.run(
                self.update_neurons,
                {
                    self.replace_from_index: more_used,
                    self.replace_to_index: less_used
                }
            )

    def get_clusters (self, vectors):
        clusters = self.train_loop.sess.run(self.batch_winners, {self.input_batch: vectors})
        self.update_stat(clusters)
        return clusters

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
