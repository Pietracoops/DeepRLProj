import yaml

from dqn_agent import DQNAgent
from environment import Environment

def import_config():
    with open("../conf/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            config = { }
            print(exc)
    return config

def run_training_loop(config, env, agent):
    for i in range(config["alg"]["n_iter"]):
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, terminal = env.update(action)
        
        agent.store(state, action, next_state, reward, terminal)
        agent.update()


config = import_config()
print("Config: {}".format(config))

env = Environment()
agent = DQNAgent(config["agent"])

run_training_loop(config, env, agent)