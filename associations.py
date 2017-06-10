import tensorflow as tf
import numpy as np
import cv2
from som.som import SOM

class Associations ():

    def __init__ (self, sess, graph):

        self.sess = sess
        self.graph = graph

        self.state_size = 50

        self.som = SOM(20, 20, self.state_size, 1, graph=graph, sess=sess)

    def get_train_op (self, states):

        self.som.create_model (states)
        return self.som.get_train_op ()
        # self.som.train (states)

        # image_grid = np.array(self.som.get_centroids())
        # print ('centroids')
        # print (image_grid.shape)
        # print (image_grid)
        # cv2.imshow('image', cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey (1)
