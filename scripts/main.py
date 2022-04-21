import time
import yaml

from dqn_agent import DQNAgent
from environment import Environment
from logger import Logger

def import_config():
    with open("../conf/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            config = { }
            print(exc)
    return config

def run_training_loop(config, env, agent, logger):
    #state = env.get_state()
    #action = agent.get_action(state)
    #print("Action: {}".format(action))
    
    for i in range(config["alg"]["n_iter"]):
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, terminal = env.update(action)
        
        if terminal == 1.0:
            env.reset()
        
        agent.store(state, action, next_state, reward, terminal)
        loss = agent.update()

        data = { }
        data["action"] = action
        data["reward"] = reward
        data["terminal"] = terminal
        data["loss"] = loss
        logger.log(data)
        
config = import_config()
print("Config: {}".format(config))

env = Environment(config["env"])
agent = DQNAgent(config["alg"]["n_iter"], config["agent"])
logger = Logger(config, "../log/" + time.strftime("%d-%m-%Y_%H-%M-%S"))

run_training_loop(config, env, agent, logger)