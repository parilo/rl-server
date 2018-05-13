import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape, Flatten, Dropout, Conv2D, MaxPooling2D

class SnakeModel(object):
    def __init__(self, input_shapes, output_size, model=None, scope=None):

        self._input_shapes = input_shapes
        self._output_size = output_size
        self._scope = scope or 'SnakeModel'

# snake_agent = DQNAgent(env, num_actions, state_shape=[8, 8, 5],
#                        convs=[[16, 2, 1], [32, 1, 1]], fully_connected=[128],
#                        save_path="snake_models", model_name="dqn_8x8")

        input_shape = input_shapes[0]

        is_actor = len(input_shapes) == 1
        is_critic = len(input_shapes) == 2

        if is_critic:
            critic_input_shape = input_shapes[1]

        if model is None:

            state_input = keras.layers.Input(shape=input_shape, name='state_input_dummy')

            with tf.variable_scope(self._scope):

                if is_actor:
                    pass

                elif is_critic:

                    critic_input = keras.layers.Input(shape=critic_input_shape, name='critic_input_dummy')

                    conv_network = Sequential ([
                        Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=input_shape),
                        Conv2D(filters=32, kernel_size=1, strides=1, activation='relu'),
                        Flatten()
                    ])

                    joined = keras.layers.Concatenate(axis=1)([
                        conv_network(state_input),
                        critic_input
                    ])

                    ff_input_shape = (int(joined.shape[1]),)

                    print('ff_input_shape: {}'.format(ff_input_shape))

                    ff_network = Sequential ([
                        Dense(128, activation='relu', input_shape=ff_input_shape),
                        Dense(self._output_size)
                    ])

                    model_inputs = [
                        state_input,
                        critic_input
                    ]

                    self._model = keras.models.Model(inputs=model_inputs, outputs=ff_network(joined))

        else:
            self._model = model

    def __call__(self, xs):

        # xs may be (input_1, input_2, ...)
        # or ((input_1, input_2, ...), additional_input)

        if isinstance(xs[0], (list, tuple)):
            print('--- call 1: {}'.format(xs))
            return self._model([xs[0][0], xs[1]])
        else:
            print('--- call 2: {}'.format(xs))
            return self._model(xs)

    def variables(self):
        return self._model.trainable_weights

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope) as sc:
            m = keras.models.model_from_json(self._model.to_json())
            m.set_weights(self._model.get_weights())
            return SnakeModel(
                self._input_shapes,
                self._output_size,
                model=m,
                scope=scope
            )
