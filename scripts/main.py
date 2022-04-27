import gc
import sys
import time
import yaml

import torch

import utils

from dqn_agent import DQNAgent
from ddpg_agent import DDPGAgent
from environment import Environment
from logger import Logger

def import_config(option):
    path = "../conf/config.yaml"
    if option == '1':
        path = "../conf/test_config.yaml"

    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            config = { }
            print(exc)
    return config

def run_training_loop(config, env, agent, logger):
    for i in range(config["alg"]["n_iter"]):
        print("\n\n********** Iteration %i ************"%i)

        state = env.current_state
        action = agent.get_action(state)
        next_state, reward, terminal = env.update(action)

        agent.store(state, action, next_state, reward, terminal)
        update_logs = agent.update()

        data = { }
        data["action"] = action
        data["reward"] = reward
        data["terminal"] = terminal
        data["update_logs"] = update_logs
        data["collisions"] = env.collisions
        data["grasps"] = env.grasps
        logger.log(data)

        if terminal == 1.0:
            env.reset()

        del state
        del action
        del next_state
        del reward
        del terminal
        del data
        torch.cuda.empty_cache()

        if i % config["alg"]["flush_frequency"] == 0:
            gc.collect()

option = None
if len(sys.argv) > 1:
    option = sys.argv[1]

config = import_config(option)
print("Config: {}".format(config))

utils.set_device(config)

env = Environment(config)

if config["alg"]["agent"] == "dqn":
    agent = DQNAgent(config["alg"]["n_iter"], config["dqn_agent"])
elif config["alg"]["agent"] == "ddpg":
    agent = DDPGAgent(config["alg"]["n_iter"], config["ddpg_agent"])
else:
    raise NotImplementedError 

logger = Logger(config, agent, "../log/" + time.strftime("%d-%m-%Y_%H-%M-%S"))

if option is not None:
    run_training_loop(config, env, agent, logger)