# code adopted from
# https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/

import tensorflow as tf
import numpy as np
import cv2
import time
from som.som import SOM

#For plotting the images
from matplotlib import pyplot as plt

#Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.get_default_graph ()
sess = tf.Session(config=config)

#Train a 20x30 SOM with 400 iterations
som = SOM(16, 16, 3, 1, graph=graph, sess=sess)

init_op = tf.global_variables_initializer()
sess.run(init_op)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
# plt.show()

for i in range (5000):

    # batch = colors[np.random.choice(colors.shape[0], 4, replace=False), :]
    batch = np.random.uniform (0, 1, (16, 3))
    som.train(batch)

    #Get output grid
    image_grid = np.array(som.get_centroids())
    # print (image_grid.shape)
    # print (image_grid)
    cv2.imshow('image', cv2.resize(image_grid, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey (1)
    # time.sleep (0.1)

    #Map colours to their closest neurons
    # mapped = som.map_vects(colors)

    #Plot
    # plt.imshow(image_grid)
    # ax.imshow (image_grid)
    # plt.title('Color SOM')
    # for i, m in enumerate(mapped):
    #     plt.text(
    #         m[1],
    #         m[0],
    #         color_names[i],
    #         ha='center',
    #         va='center',
    #         bbox=dict(facecolor='white', alpha=0.5, lw=0)
    #     )
    # plt.show()
    # plt.draw()
    # accept = input('OK? ')

# for fname in os.listdir(dir):
#     fname = os.path.join(dir, fname)
#     im = plt.imread(fname)
#     img = ax.imshow(im)
#     plt.draw()
#     accept = raw_input('OK? ')

# cv2.waitKey(0)
cv2.destroyAllWindows()
