# RL Server

[![Quadrotor 2D RL video](http://img.youtube.com/vi/acQJfkgeiZc/0.jpg)](https://www.youtube.com/watch?v=acQJfkgeiZc)

Reinforcement Learning Server. Includes:

* DQN: Deep Q Networks [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
* DDPG: Deep Deterministic Policy Gradient [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

DQN and DDPG implemetations are taken from https://github.com/nivwusquorum/tensorflow-deepq. Thank you Szymon Sidor ;)

# Running

Currently works with [Quadrotor 2D Simulator](https://github.com/parilo/quadrotor2d-simulator) and [Learning To Run](https://github.com/stanfordnmbl/osim-rl). So you need it up nd running to see how it works. You need Python 3. It communicates with simulator with [websockets](https://websockets.readthedocs.io/en/stable/intro.html) and json.

```
git clone this repository
python main_..._.py
```

Solve python deps ;) then run again
