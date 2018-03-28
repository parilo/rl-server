#!/usr/bin/env python

import asyncio
import websockets
import json
from rl_train_loop import RLTrainLoop
from osim_rl_ddpg import OsimRL

num_actions = 12
observation_size = 3 * 8 * 8

train_loop = RLTrainLoop (num_actions, observation_size)
osim_rl = OsimRL (train_loop)

train_loop.set_loss_op (osim_rl.get_loss_op ())
train_loop.add_train_ops (osim_rl.get_train_ops ())
train_loop.init_vars ()

async def agent_connection(websocket, path):
    while websocket.open:
        req_json = await websocket.recv()
        req = json.loads(req_json)

        method = req ['method']
        if method == 'act':
            action = osim_rl.act (req ['state'])
            await websocket.send(json.dumps(action))
        elif method == 'act_batch':
            actions = osim_rl.act_batch (req ['states'])
            await websocket.send(json.dumps(actions))
        elif method == 'store_exp_batch':
            train_loop.store_exp_batch (
                req ['rewards'],
                req ['actions'],
                req ['prev_states'],
                req ['next_states']
            )
            await websocket.send('')

train_loop.train ()

start_server = websockets.serve(agent_connection, '0.0.0.0', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
train_loop.join ()
