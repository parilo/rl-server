import numpy as np

class RActDiscrete (object):

    def __init__ (self, train_loop):

        self.act_count = 4
        self.actions = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0]
        ])

    def init(self):
        pass

    def process_outputs (self):
        pass

    def get_train_ops (self):
        return []

    def get_clusters (self, actions):
        return np.argmax(np.array(actions), axis=1).tolist()

    def get_clusters_count(self):
        return self.act_count

    def get_neurons(self):
        return self.actions
