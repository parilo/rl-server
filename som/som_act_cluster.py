import tensorflow as tf
import numpy as np
import cv2
from som import SOM

class SOMActCluster ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.map_side_size = 20
        self.input_dim = train_loop.num_actions

        self.som = SOM(
            (self.input_dim,),
            self.map_side_size,
            1,
            train_loop.sess,
            train_loop.dequeued_actions
        )

    def get_train_ops (self):
        return [self.som.get_train_op (), self.som.get_centroids_op ()]

    def process_train_outputs (self):
        # print (self.train_loop.train_outputs [12].shape)
        image_grid = np.concatenate(
            [
                np.reshape(
                    self.train_loop.train_outputs [12],
                    [self.map_side_size, self.map_side_size, self.input_dim]
                ),
                np.zeros((self.map_side_size, self.map_side_size, 1))
            ],
            axis=2
        )
        # print (image_grid.shape)
        cv2.imshow('Act clusters', cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey (1)
