from tensorboardX import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, config, log_dir):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        
        self.t = 0
        self.n = config["logger"]["average_over"]
        self.logging_starts = config["agent"]["learning_starts"]
        
        self.agent_rewards = []
        self.agent_cumulative_rewards = 0

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)
        
    def log(self, data):
        self.agent_cumulative_rewards += data["reward"]
        if data["terminal"] == 1.0:
            self.agent_rewards.append(self.agent_cumulative_rewards)
            self.agent_cumulative_rewards = 0.0
            
        if len(self.agent_rewards) > 5 * self.n:
            index = self.n * 2
            self.agent_rewards[-index:]
            
        if self.t > self.logging_starts:
            logs = { } 
            logs["Average_Rewards"] = np.mean(np.array(self.agent_rewards[-self.n:]))
            logs["Current_Cumulative_Reward"] = self.agent_cumulative_rewards
            if len(self.agent_rewards) > 1:
                logs["Last_Cumulative_Reward"] = self.agent_rewards[-1]
            logs["Loss"] = data["loss"]
            
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.log_scalar(value, key, self.t)
            print('Done logging...\n\n')
            
            self.flush()
        
        self.t += 1
        
    def flush(self):
        self._summ_writer.flush()