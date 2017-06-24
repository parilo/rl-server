import tensorflow as tf
import numpy as np
import cv2
from som import SOM

class SOMCluster ():

    def __init__ (
        self,
        train_loop,
        map_side_size,
        input_dim,
        samples_tensor,
        map_title
    ):
        self.train_loop = train_loop
        self.map_side_size = map_side_size
        self.input_dim = input_dim
        self.map_title = map_title
        self.highlighted_vector = np.zeros((1, self.input_dim))

        self.som = SOM(
            (self.input_dim,),
            self.map_side_size,
            1,
            train_loop.sess,
            samples_tensor
        )

    def get_clusters_count (self):
        return self.map_side_size * self.map_side_size

    def get_train_ops (self):
        return [self.som.get_train_op (), self.som.get_centroids_op ()]

    def get_clusters (self, vectors):
        return self.som.get_batch_winner (vectors)

    def show_centroids (self, image_grid):
        #normalization of centroids
        # centroids = self.train_loop.train_outputs [self.centroid_train_output_index]
        # max0 = np.max(centroids[:,0])
        # min0 = np.min(centroids[:,0])
        # max1 = np.max(centroids[:,1])
        # min1 = np.min(centroids[:,1])
        # centroids[:,0] = (centroids[:,0] - min0) / (max0 - min0)
        # centroids[:,1] = (centroids[:,1] - min1) / (max1 - min1)

        # image_grid = np.concatenate(
        #     [
        #         np.reshape(
        #             self.train_loop.train_outputs [self.centroid_train_output_index], # get cendroids op of this self organized map
        #             [self.map_side_size, self.map_side_size, self.input_dim]
        #         ),
        #         np.zeros((self.map_side_size, self.map_side_size, 1))
        #     ],
        #     axis=2
        # )

        cmax = np.max(image_grid)
        cmin = np.min(image_grid)
        image_grid = (image_grid - cmin) / (cmax - cmin)

        index = self.som.get_batch_winner (self.highlighted_vector) [0]
        highlighted = divmod(index, self.map_side_size)
        image_grid [highlighted[0], highlighted[1], :] = (1, 1, 1)

        # print (image_grid)
        cv2.imshow(self.map_title, cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey (1)

    def highlight (self, vector):
        self.highlighted_vector = vector.reshape((1, -1))
        # print (act)
        # print (self.highlighted)
