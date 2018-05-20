#!/usr/bin/env python

import asyncio
import websockets
import json
import tensorflow as tf
from rl_train_loop import RLTrainLoop
from snake_dqn import SnakeDQN

# num_actions = 3
observation_shapes = [[8, 8, 5]]
# observation_shapes = [[8 * 8 * 5]]

# snake_agent = DQNAgent(env, num_actions, state_shape=[8, 8, 5],
#                        convs=[[16, 2, 1], [32, 1, 1]], fully_connected=[128],
#                        save_path="snake_models", model_name="dqn_8x8")
#
# snake_agent.set_parameters(max_episode_length=1000, replay_memory_size=100000, replay_start_size=10000,
#                            discount_factor=0.999, final_eps=0.01, annealing_steps=100000)
#
# def train(self,
#           gpu_id=0,
#           batch_size=32,
#           exploration="e-greedy",
#           agent_update_freq=4,
#           target_update_freq=5000,
#           tau=1,
#           max_num_epochs=50000,
#           performance_print_freq=500,
#           save_freq=10000,
#           from_epoch=0):
#
# class DeepQNetwork:
#
#     def __init__(self, num_actions, state_shape=[8, 8, 5],
#                  convs=[[32, 4, 2], [64, 2, 1]],
#                  fully_connected=[128],
#                  optimizer=tf.train.AdamOptimizer(2.5e-4),
#                  activation_fn=tf.nn.relu,
#                  scope="dqn", reuse=False):

train_loop = RLTrainLoop (
    observation_shapes,
    action_size = 1,
    action_dtype = tf.int32,
    is_actions_space_continuous = False,
    batch_size = 32,
    discount_rate = 0.999,
    experience_replay_buffer_size = 100000,
    store_every_nth = 1,
    start_learning_after = 10000
)

algorithm = SnakeDQN (train_loop)

train_loop.set_loss_op (algorithm.get_loss_op ())
train_loop.add_train_ops (algorithm.get_train_ops ())
train_loop.init_vars ()

async def agent_connection(websocket, path):
    while websocket.open:
        req_json = await websocket.recv()
        req = json.loads(req_json)

        method = req ['method']
        if method == 'act':
            scores, action = algorithm.act (req ['state'])
            await websocket.send(json.dumps({
                "action" : action,
                "scores" : scores
            }))
        elif method == 'act_batch':
            actions = algorithm.act_batch (req ['states'])
            await websocket.send(json.dumps(actions))
        elif method == 'store_exp_batch':
            train_loop.store_exp_batch (
                req ['rewards'],
                req ['actions'],
                req ['prev_states'],
                req ['next_states'],
                req ['terminator'],
            )
            await websocket.send('')

train_loop.train ()

start_server = websockets.serve(agent_connection, '0.0.0.0', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
train_loop.join ()
