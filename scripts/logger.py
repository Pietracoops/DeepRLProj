from tensorboardX import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, config, agent, log_dir):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        
        self.t = 0
        self.n = config["logger"]["average_over"]
        self.logging_starts = agent.learning_starts
        self.agent = config["alg"]["agent"]
        
        self.agent_rewards = []
        self.agent_cumulative_rewards = 0

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def flush(self):
        self._summ_writer.flush()

    def log(self, data):
        self.agent_cumulative_rewards += data["reward"]
        if data["terminal"] == 1.0:
            self.agent_rewards.append(self.agent_cumulative_rewards)
            self.agent_cumulative_rewards = 0.0
            
        if len(self.agent_rewards) > 5 * self.n:
            index = self.n * 2
            self.agent_rewards[-index:]

        if self.agent == "dqn":
            logs = self.log_dqn(data)
        elif self.agent == "ddpg":
            logs = self.log_ddpg(data)
        else:
            raise NotImplementedError 

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.log_scalar(value, key, self.t)
        print('Done logging...\n\n')
            
        self.flush()
        
        self.t += 1

        
    def log_dqn(self, data):            
        logs = { } 
        logs["Current_Cumulative_Reward"] = self.agent_cumulative_rewards
        if self.t > self.logging_starts:
            if len(self.agent_rewards) > 1:
                logs["Average_Rewards"] = np.mean(np.array(self.agent_rewards[-self.n:]))
                logs["Last_Cumulative_Reward"] = self.agent_rewards[-1]
            if data["update_logs"] is not None:
                logs["Push_Network_Loss"] = data["update_logs"]["push_network"]["loss"]
                logs["Push_Network_Q_Values"] = data["update_logs"]["push_network"]["q_values"]
                logs["Push_Network_Q_Targets"] = data["update_logs"]["push_network"]["target_q_values"]

                logs["Grasp_Network_Loss"] = data["update_logs"]["grasp_network"]["loss"]
                logs["Grasp_Network_Q_Values"] = data["update_logs"]["grasp_network"]["q_values"]
                logs["Grasp_Network_Q_Targets"] = data["update_logs"]["grasp_network"]["target_q_values"]

        return logs

    def log_ddpg(self, data):
        logs = { } 
        logs["Current_Action"] = data["action"].item()
        logs["Current_Cumulative_Reward"] = self.agent_cumulative_rewards
        logs["Collisions"] = data["collisions"]
        logs["Graps"] = data["graps"]
        if self.t > self.logging_starts:
            if len(self.agent_rewards) > 1:
                logs["Average_Rewards"] = np.mean(np.array(self.agent_rewards[-self.n:]))
                logs["Last_Cumulative_Reward"] = self.agent_rewards[-1]
            if data["update_logs"] is not None:
                logs["Critic_Loss"] = data["update_logs"]["critic"]["loss"]
                logs["Critic_Q_Values"] = data["update_logs"]["critic"]["q_values"]
                logs["Critic_Q_Targets"] = data["update_logs"]["critic"]["target_q_values"]

                logs["Actor_Loss"] = data["update_logs"]["actor"]["loss"]
                logs["Actor_Actions"] = data["update_logs"]["actor"]["actions"]

        return logs