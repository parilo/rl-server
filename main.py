#!/usr/bin/env python

#https://websockets.readthedocs.io/en/stable/intro.html

import tensorflow as tf
import asyncio
import websockets
import json
from quadrotor2d import Quadrotor2D
from associations import Associations

model = Quadrotor2D ()
sess = model.sess
associations = Associations (sess, model.graph)
assoc_train_op = associations.get_train_op (model.q_next_states)
sess.run(tf.global_variables_initializer())

async def agent_connection(websocket, path):
    while websocket.open:
        req_json = await websocket.recv()
        req = json.loads(req_json)
        # print (req)

        method = req ['method']
        if method == 'act':
            action = model.act (req ['state'])
            await websocket.send(json.dumps(action))
        elif method == 'act_batch':
            actions = model.act_batch (req ['states'])
            await websocket.send(json.dumps(actions))
        elif method == 'store_exp_batch':
            model.store_exp_batch (
                req ['rewards'],
                req ['actions'],
                req ['prev_states'],
                req ['next_states']
            )
            await websocket.send('')

model.train ()

start_server = websockets.serve(agent_connection, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
model.join ()
