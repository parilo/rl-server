import numpy as np
from .r_cluster import RCluster

class RActCluster (RCluster):

    def __init__ (self, train_loop):

        # a = 10
        # mean_state_val = np.array([a, a])
        # std_state_val = np.array([a, a])

        mean_state_val = np.array([0.0, 0, 0, 0])
        std_state_val = np.array([1.0, 1.0, 1.0, 1.0])

        super().__init__(
            train_loop,
            5, # map_side_size,
            # (0.0, 15.0), # amplitude
            mean_state_val,
            std_state_val,
            train_loop.num_actions, # input_dim,
            # train_loop.dequeued_actions, # samples_tensor,
            train_loop.inp_actions,
            'act_clusters' # scope
        )

    def process_outputs (self):
        # self.show_centroids (None)
        pass

        # centroids = self.train_loop.store_outputs [8]
        # self._set_centroids (centroids)
        #
        # image_grid = np.concatenate(
        #     [
        #         np.reshape(
        #             centroids, # get cendroids op of this self organized map
        #             [self.map_side_size, self.map_side_size, self.input_dim]
        #         ),
        #         np.zeros((self.map_side_size, self.map_side_size, 1))
        #     ],
        #     axis=2
        # )
        #
        # self.show_centroids (image_grid)
