import gc
import time
import yaml

import torch

import utils

from dqn_agent import DQNAgent
from environment import Environment
from logger import Logger

def import_config():
    #with open("../conf/test_config.yaml", "r") as stream:
    with open("../conf/config.yaml", "r") as stream:
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
        logger.log(data)

        if terminal == 1.0:
            env.reset()

        if i % config["alg"]["flush_frequency"] == 0:
            gc.collect()
            torch.cuda.empty_cache()

config = import_config()
print("Config: {}".format(config))

utils.set_device(config)

env = Environment(config["env"])
agent = DQNAgent(config["alg"]["n_iter"], config["agent"])
logger = Logger(config, "../log/" + time.strftime("%d-%m-%Y_%H-%M-%S"))

run_training_loop(config, env, agent, logger)