import numpy as np
from .r_cluster import RCluster

class RStateCluster (RCluster):

    def __init__ (self, train_loop):

        super().__init__(
            train_loop,
            20, # map_side_size,
            (-10.0, 10.0), # amplitude
            train_loop.observation_size, # input_dim,
            # train_loop.dequeued_next_states, # samples_tensor,
            train_loop.inp_next_states,
            'state_clusters' # scope
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

    def process_outputs (self):
        # image_grid = np.reshape(self.train_loop.store_outputs [6], [self.map_side_size, self.map_side_size, self.input_dim])[:,:,5:8]
        # self.show_centroids (image_grid)
        pass
