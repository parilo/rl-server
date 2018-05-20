import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape, Flatten, Dropout, Conv2D, MaxPooling2D

class SnakeModelDense(object):
    def __init__(self, input_shapes, output_size, model=None, scope=None):

        self._input_shapes = input_shapes
        self._output_size = output_size
        self._scope = scope or 'SnakeModelDense'

        # r = tf.nn.relu
        # t = tf.nn.tanh
        # critic = MLP([observation_shapes[0][0], num_actions], [256, 256, 256, 256, 256, 1],
        #             [r, r, r, t, t, tf.identity], scope='critic')

        input_shape = input_shapes[0]

        if model is None:

            state_input = keras.layers.Input(shape=input_shape, name='state_input_dummy')

            with tf.variable_scope(self._scope):

                ff_network = Sequential ([
                    Dense(256, activation='relu', input_shape=input_shape),
                    Dense(256, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(256, activation='tanh'),
                    Dense(256, activation='tanh'),
                    Dense(self._output_size)
                ])

                model_inputs = [
                    state_input
                ]

                self._model = keras.models.Model(inputs=model_inputs, outputs=ff_network(state_input))

        else:
            self._model = model

    def __call__(self, xs):
        return self._model(xs)

    def variables(self):
        return self._model.trainable_weights

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope) as sc:
            m = keras.models.model_from_json(self._model.to_json())
            m.set_weights(self._model.get_weights())
            return SnakeModelDense(
                self._input_shapes,
                self._output_size,
                model=m,
                scope=scope
            )
