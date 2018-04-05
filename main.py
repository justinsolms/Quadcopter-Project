"""Main run."""

import getopt
from glob import glob
import os
import sys
import pandas as pd
from agents.ddpg import DDPG
from task import Task
import numpy as np
# import matplotlib.pyplot as plt


def train(num_episodes=10000, ):
    """Train."""
    telemetry = list()

    target_pos = np.array([0., 0., 10.])
    # Create the task environment.
    task = Task(target_pos=target_pos, runtime=100.0)
    # Create the DDPG agent in the task environment.
    agent = DDPG(task)

    for i_episode in range(1, num_episodes+1):
        # start a new episode
        state = agent.reset_episode()
        episode_reward = 0.0
        N = 0
        while True:
            # Actor commands the action
            action = agent.act(state)
            # Environment reacts with next state, reward and done for
            # end-of-episode
            next_state, reward, done = task.step(action)
            # Agent (actor-critic) learns
            losses = agent.step(action, reward, next_state, done)
            # S <- S
            state = next_state
            episode_reward += reward
            N += 1
            if done and losses is not None:
                loss_critic = losses
                # End of episode. Show metrics.
                print('\rEpisode: {:4d}, Loss-crit: {:8.3f}, Av Rwd: {:7.3f}, Lst Rwd: {:6.3f}, RunTime: {:6.3f}'.format(i_episode, loss_critic, episode_reward/N, reward, task.sim.time))
                telemetry.append((i_episode, loss_critic, episode_reward))
            if done:
                break
        # Re-use same line to print on.
        sys.stdout.flush()

    # Plot
    i_episode, loss_actor, loss_critic = zip(*telemetry)
    # plt.plot(i_episode, loss_actor)
    # plt.plot(i_episode, loss_critic)
    # plt.show()

def main(argv):
    """Command line parser for batch processes."""
    # FIXME: This is a sample from another project.
    longopts = ['job=', 'batch_size=', 'z_dim=', 'learning_rate=', 'beta1=']
    try:
        opts, args = getopt.getopt(argv, '', longopts)
    except getopt.GetoptError:
        print('Bad input arguments.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--job':
            job = arg
        elif opt == '--batch_size':
            batch_size = int(arg)
        elif opt == '--z_dim':
            z_dim = int(arg)
        elif opt == '--learning_rate':
            learning_rate = float(arg)
        elif opt == '--beta1':
            beta1 = float(arg)

    msg = ('Train %s\n'
           '  batch_size = %i \n'
           '  z_dim = %i \n'
           '  learning_rate = %f \n'
           '  beta1 = %f \n') % (job, batch_size, z_dim, learning_rate, beta1)
    print(msg)

    if job == 'mnist':
        """
        DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
        """
        epochs = 2

        mnist_dataset = helper.Dataset('mnist', glob(
            os.path.join(data_dir, 'mnist/*.jpg')))
        with tf.Graph().as_default():
            train(epochs, batch_size, z_dim, learning_rate, beta1,
                  mnist_dataset.get_batches, mnist_dataset.shape,
                  mnist_dataset.image_mode, 'mnist')

    elif job == 'celeba':

        """
        DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
        """
        epochs = 2

        celeba_dataset = helper.Dataset('celeba', glob(
            os.path.join(data_dir, 'img_align_celeba/*.jpg')))
        with tf.Graph().as_default():
            train(epochs, batch_size, z_dim, learning_rate, beta1,
                  celeba_dataset.get_batches, celeba_dataset.shape,
                  celeba_dataset.image_mode, 'celeba')


if __name__ == '__main__':
    main(sys.argv[1:])
