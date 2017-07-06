import numpy as np
from .som_cluster import SOMCluster

class SOMActCluster (SOMCluster):

    def __init__ (self, train_loop):

        super().__init__(
            train_loop,
            20, # map_side_size,
            train_loop.num_actions, # input_dim,
            # train_loop.dequeued_actions, # samples_tensor,
            train_loop.inp_actions,
            'Act clusters' # map_title
        )

    def process_train_outputs (self):

        centroids = self.train_loop.train_outputs [16]
        self._set_centroids (centroids)

        image_grid = np.concatenate(
            [
                np.reshape(
                    centroids, # get cendroids op of this self organized map
                    [self.map_side_size, self.map_side_size, self.input_dim]
                ),
                np.zeros((self.map_side_size, self.map_side_size, 1))
            ],
            axis=2
        )

        self.show_centroids (image_grid)
