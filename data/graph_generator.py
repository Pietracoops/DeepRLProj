import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Script for plot generation from tensorboard event files

# Example inspired from https://www.programcreek.com/python/example/114903/tensorboard.backend.event_processing.event_accumulator.EventAccumulator


def make_plots(input, iteration_max):

    dirname = os.path.dirname(__file__)
    _size = 100

    Q1_folder = os.path.join(dirname, '27-04-2022_10-42-10')

    # Experiment 1
    Q1_event_file1 = os.path.join(Q1_folder, 'events.out.tfevents.1651070530.LAPTOP-ULCIRELK')


    size = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': _size,
        'histograms': 1
    }

    if input == 1:
        # Include a learning curve plot showing the performance of your implementation on Ms. Pac-Man. The x-axis should correspond to number of time steps
        # (consider using scientific notation) and the y-axis should show the average per-epoch reward as well as the
        # best mean reward so far. These quantities are already computed and printed in the starter code. They are
        # also logged to the data folder, and can be visualized using Tensorboard as in previous assignments. Be sure to
        # label the y-axis, since we need to verify that your implementation achieves similar reward as ours. You should
        # not need to modify the default hyperparameters in order to obtain good performance, but if you modify any
        # of the parameters, list them in the caption of the figure

        print("Loading Graph Set 1")
        p1 = EventAccumulator(Q1_event_file1, size); p1.Reload(); p1_plot = plt # mspacman

        Current_action_p1 = p1.Scalars('Current_Action')
        last_cum_reward_p1 = p1.Scalars('Last_Cumulative_Reward')
        Train_AverageReturn_p1 = p1.Scalars('Average_Rewards')
        Best_AverageReturn_p1 = p1.Scalars('Current_Cumulative_Reward')
        critic_loss_EventsReturn_p1 = p1.Scalars('Critic_Loss')
        Critic_Q_vals_EventsReturn_p1 = p1.Scalars('Critic_Q_Values')
        Critic_Q_targets_EventsReturn_p1 = p1.Scalars('Critic_Q_Targets')
        Actor_loss_EventsReturn_p1 = p1.Scalars('Actor_Loss')
        Actor_Actions_EventsReturn_p1 = p1.Scalars('Actor_Actions')
        Collisions_p1 = p1.Scalars('Collisions')
        Grasps_p1 = p1.Scalars('Grasps')




        iterations = np.zeros([len(Current_action_p1),1])
        Current_action_x_axis = np.zeros([len(Current_action_p1),1])
        last_cum_reward_x_axis = np.zeros([len(last_cum_reward_p1), 1])
        Train_AverageReturn_x_axis = np.zeros([len(Train_AverageReturn_p1), 1])
        Best_AverageReturn_x_axis = np.zeros([len(Best_AverageReturn_p1), 1])
        critic_loss_EventsReturn_x_axis = np.zeros([len(critic_loss_EventsReturn_p1), 1])
        Critic_Q_vals_EventsReturn_x_axis = np.zeros([len(Critic_Q_vals_EventsReturn_p1), 1])
        Critic_Q_targets_EventsReturn_x_axis = np.zeros([len(Critic_Q_targets_EventsReturn_p1), 1])
        Actor_loss_EventsReturn_x_axis = np.zeros([len(Actor_loss_EventsReturn_p1), 1])
        Actor_Actions_EventsReturn_x_axis = np.zeros([len(Actor_Actions_EventsReturn_p1), 1])

        y_axis = np.zeros([len(Current_action_p1), 11])

        iteration_interval = int(float(iteration_max) / float(_size))

        for i in range(len(Train_AverageReturn_p1)):
            iterations[i, 0] = i * iteration_interval
            # Current_action_x_axis[i, 0] = Current_action_p1[i][2]
            # last_cum_reward_x_axis[i, 0] = last_cum_reward_p1[i][2]
            # Train_AverageReturn_x_axis[i, 0] = Train_AverageReturn_p1[i][2]
            # Best_AverageReturn_x_axis[i, 0] = Best_AverageReturn_p1[i][2]
            # critic_loss_EventsReturn_x_axis[i, 0] = critic_loss_EventsReturn_p1[i][2]
            # Critic_Q_vals_EventsReturn_x_axis[i, 0] = Critic_Q_vals_EventsReturn_p1[i][2]
            # Critic_Q_targets_EventsReturn_x_axis[i, 0] = Critic_Q_targets_EventsReturn_p1[i][2]
            # Actor_loss_EventsReturn_x_axis[i, 0] = Actor_loss_EventsReturn_p1[i][2]
            # Actor_Actions_EventsReturn_x_axis[i, 0] = Actor_Actions_EventsReturn_p1[i][2]
            y_axis[i, 0] = Current_action_p1[i][2]
            y_axis[i, 1] = last_cum_reward_p1[i][2]
            y_axis[i, 2] = Train_AverageReturn_p1[i][2]
            y_axis[i, 3] = Best_AverageReturn_p1[i][2]
            y_axis[i, 4] = critic_loss_EventsReturn_p1[i][2]
            y_axis[i, 5] = Critic_Q_vals_EventsReturn_p1[i][2]
            y_axis[i, 6] = Critic_Q_targets_EventsReturn_p1[i][2]
            y_axis[i, 7] = Actor_loss_EventsReturn_p1[i][2]
            y_axis[i, 8] = Actor_Actions_EventsReturn_p1[i][2]
            y_axis[i, 9] = Collisions_p1[i][2]
            y_axis[i, 10] = Grasps_p1[i][2]

        # Graph 1
        p1_plot.plot(iterations[:, 0], y_axis[:, 0], label='Current Action')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Action")
        p1_plot.title("Current Action of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 2
        p1_plot.plot(iterations[:, 0], y_axis[:, 1], label='Last Cumulative Reward')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Last Cumulative Reward")
        p1_plot.title("Last Cumulative Reward of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 3
        p1_plot.plot(iterations[:, 0], y_axis[:, 2], label='Train Average Return')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Average Return")
        p1_plot.title("Train Average Return of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 4
        p1_plot.plot(iterations[:, 0], y_axis[:, 3], label='Best Average Return')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Average Return")
        p1_plot.title("Best Average Return of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 5
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Critic Loss')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Loss")
        p1_plot.title("Critic Loss of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 6
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Critic Q Vals')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Q Vals")
        p1_plot.title("Critic Q Values of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 7
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Critic Q Targets')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Q Vals")
        p1_plot.title("Critic Q Targets of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 8
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Actor Loss')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Loss")
        p1_plot.title("Actor Loss of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 9
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Actor Actions')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Action")
        p1_plot.title("Actor Actions of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 10
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Collisions')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Collisions")
        p1_plot.title("Collisions of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()

        # Graph 11
        p1_plot.plot(iterations[:, 0], y_axis[:, 4], label='Grasps')
        p1_plot.xlabel("Iteration")
        p1_plot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        p1_plot.xticks(np.arange(0, iterations[len(Train_AverageReturn_p1) - 1, 0], step=iterations[len(Train_AverageReturn_p1) - 1, 0] / 10))
        p1_plot.ylabel("Grasps")
        p1_plot.title("Grasps of Robotic Arm per Iteration")
        p1_plot.legend(loc='best', frameon=True)
        p1_plot.show()



if __name__ == '__main__':
    print("script started")

    print("NOTE: Files are extremely large and make take time to load them, please be patient")

    print("Generating Plots for Experiment 1")
    iteration_max = 12920
    make_plots(1, iteration_max)

    print("Terminated")
