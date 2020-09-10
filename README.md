# Deep reinforcement learning in PyTorch
This repo contains my project work for the Autonomous and Adaptive Systems course taught by professor Mirco Musolesi at the University of Bologna.

## Agents
The project contains:
* A DQN agent
* A Double DQN agent
* An Actor-Critic agent

Planning to implement:
* Prioritized experience replay
* Dueling DQN

To train the agents you can run these main files:
* [dqn_main.py](https://github.com/alessandropacielli/autonomous_systems_project/blob/master/dqn_main.py)
* [double_dqn_main.py](https://github.com/alessandropacielli/autonomous_systems_project/blob/master/double_dqn_main.py)
* [actor_critic_main.py](https://github.com/alessandropacielli/autonomous_systems_project/blob/master/actor_critic_main.py)

( A nice CLI is another TODO on the list :) )

## Dependencies
The dependencies are managed with [Poetry](https://python-poetry.org/), to install them first initialize a virtualenv and activate it:
```
python3 -m venv .venv
source .venv/bin/activate
```
Then install poetry with:
```
pip install poetry
```
Then run:
```
poetry install
```
