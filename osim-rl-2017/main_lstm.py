#!/usr/bin/env python

import subprocess
import atexit
import time

#
# Probably will not work
# I left it here because it may be useful
#

ps = []

for i in range (2):
    ps.append(subprocess.Popen(['python', 'agent_lstm.py', '--visualize']))

for i in range (3):
    ps.append(subprocess.Popen(['python', 'agent_lstm.py']))

def on_exit():
    for p in ps:
        p.kill ()

atexit.register(on_exit)

while True:
    time.sleep(60)
