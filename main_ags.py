#!/usr/bin/env python

#https://websockets.readthedocs.io/en/stable/intro.html

import asyncio
import websockets
import json
import numpy as np
import random
from rl_train_loop import RLTrainLoop
from quadrotor2d_ddpg import Quadrotor2D
# from som.som_state_cluster import SOMStateCluster
# from som.som_act_cluster import SOMActCluster
from random_layer.r_state_cluster import RStateCluster
from random_layer.r_act_cluster import RActCluster
# from associations.associations_graph_som_clustering import AssociationsGraph
from associations.associations_graph import AssociationsGraph

# AGS - associations graph search

num_actions = 2;
observation_size = 50;

train_loop = RLTrainLoop (num_actions, observation_size)
quadrotor2d = Quadrotor2D (train_loop)
# state_cluster = SOMStateCluster (train_loop)
# act_cluster = SOMActCluster (train_loop)
state_cluster = RStateCluster (train_loop)
act_cluster = RActCluster (train_loop)
assoc = AssociationsGraph (train_loop, state_cluster, act_cluster)

train_loop.set_loss_op (quadrotor2d.get_loss_op ())
train_loop.add_train_ops (quadrotor2d.get_train_ops ()) # 3 train ops
train_loop.add_store_ops (state_cluster.get_train_ops ()) # 2 train ops
train_loop.add_store_ops (act_cluster.get_train_ops ()) # 2 train ops

def store_listener ():
    # state_cluster.process_outputs ()
    act_cluster.process_outputs ()
    assoc.process_outputs ()

train_loop.set_store_listener (store_listener)
train_loop.init_vars ()
act_cluster.init ()
state_cluster.init ()
state_cluster.set_updatable (True)

async def agent_connection(websocket, path):
    while websocket.open:
        try:
            req_json = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            print ('-------------------------------')
            print ('-------------------------------')
            print ('closed')
            print ('-------------------------------')
            print ('-------------------------------')
            train_loop.stop_train ()
            return
        req = json.loads(req_json)
        # print (req)

        method = req ['method']
        if method == 'act':
            action = quadrotor2d.act (req ['state'])
            await websocket.send(json.dumps(action))
        elif method == 'act_batch':

            # print('--- states')
            # print(req ['states'])
            actions_assoc = assoc.control (req ['states'])
            # print('--- output')
            # print(actions_assoc)
            # actions_ddpg = quadrotor2d.act_batch (req ['states'])

            # actions = None
            # if (actions_assoc is None):
            #     actions = actions_ddpg
            # else:
            #     actions = 0.5 * np.array(actions_assoc) + 0.5 * np.array(actions_ddpg)
            #     actions = actions.tolist ()

            # actions = None
            # if random.random () > 0.5:
            #     actions = assoc.control (req ['states'])
            # else:
            #     actions = quadrotor2d.act_batch (req ['states'])
            #
            # if (actions is None):
            #     actions = quadrotor2d.act_batch (req ['states'])

            # actions = quadrotor2d.act_batch (req ['states'])

            # else:
            #     actions = 0.5 * np.array(actions_assoc) + 0.5 * np.array(actions_ddpg)
            #     actions = actions.tolist ()
                # actions = quadrotor2d.act_batch (req ['states'])
                # print ('------ nn')
                # print (actions)
            # else:
            #     print (actions)

            # state_cluster.highlight (np.array(req ['states'][10]))
            # act_cluster.highlight (np.array(actions [10]))

            await websocket.send(json.dumps(actions_assoc))

        elif method == 'store_exp_batch':
            # print('--- received')
            # print(req ['actions'])
            train_loop.store_exp_batch (
                req ['rewards'],
                req ['actions'],
                req ['prev_states'],
                req ['next_states']
            )
            await websocket.send('')

# train_loop.train ()

start_server = websockets.serve(agent_connection, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
# train_loop.join ()
