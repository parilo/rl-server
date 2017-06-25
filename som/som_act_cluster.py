import numpy as np
from .som_cluster import SOMCluster

class SOMActCluster (SOMCluster):

    def __init__ (self, train_loop):

        super().__init__(
            train_loop,
            20, # map_side_size,
            train_loop.num_actions, # input_dim,
            train_loop.dequeued_actions, # samples_tensor,
            'Act clusters' # map_title
        )

    def process_train_outputs (self):

        centroids = self.train_loop.train_outputs [12]
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

    #
    #     self.train_loop = train_loop
    #     self.map_side_size = 20
    #     self.input_dim = train_loop.num_actions
    #     self.highlighted = (0, 0)
    #
    #     self.som = SOM(
    #         (self.input_dim,),
    #         self.map_side_size,
    #         1,
    #         train_loop.sess,
    #         train_loop.dequeued_actions
    #     )
    #
    # def get_train_ops (self):
    #     return [self.som.get_train_op (), self.som.get_centroids_op ()]
    #
    # def process_train_outputs (self):
    #     # print (self.train_loop.train_outputs [12].shape)
    #     image_grid = np.concatenate(
    #         [
    #             np.reshape(
    #                 self.train_loop.train_outputs [12], # get cendroids op of this self organized map
    #                 [self.map_side_size, self.map_side_size, self.input_dim]
    #             ),
    #             np.zeros((self.map_side_size, self.map_side_size, 1))
    #         ],
    #         axis=2
    #     )
    #     image_grid [self.highlighted[0], self.highlighted[1], :] = (1, 1, 1)
    #     print (image_grid)
    #     cv2.imshow('Act clusters', cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
    #     cv2.waitKey (1)
    #
    # def highlight (self, act):
    #     index = self.som.get_batch_winner (act.reshape((1, -1))) [0]
    #     self.highlighted = divmod(index, self.map_side_size)
    #     print (act)
    #     print (self.highlighted)
