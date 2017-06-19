import numpy as np
from .som_cluster import SOMCluster

class SOMStateCluster (SOMCluster):

    def __init__ (self, train_loop):

        super().__init__(
            train_loop,
            40, # map_side_size,
            train_loop.observation_size, # input_dim,
            train_loop.dequeued_next_states, # samples_tensor,
            'State clusters' # map_title
        )

    #     self.train_loop = train_loop
    #     self.map_side_size = 40
    #     self.input_dim = train_loop.observation_size
    #
    #     self.som = SOM(
    #         (self.input_dim,),
    #         self.map_side_size,
    #         1,
    #         train_loop.sess,
    #         train_loop.dequeued_next_states,
    #         alpha_learning_rate=0.01
    #     )
    #
    # def get_train_ops (self):
    #     return [self.som.get_train_op (), self.som.get_centroids_op ()]

    def process_train_outputs (self):
        # print (self.train_loop.train_outputs [10].shape)
        image_grid = np.reshape(self.train_loop.train_outputs [10], [self.map_side_size, self.map_side_size, self.input_dim])[:,:,5:8]
        self.show_centroids (image_grid)
        # cv2.imshow('State clusters', cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey (1)
