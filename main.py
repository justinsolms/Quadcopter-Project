"""Main run."""

import getopt
from glob import glob
import os
import sys
import pandas as pd
from agents.ddpg import DDPG
from task import Task
import gym
import numpy as np
import csv
import matplotlib.pyplot as plt

labels = ['episode', 'step', 'lossQ', 'reward']
results = {x : [] for x in labels}
name = 'Pendulum-v0'

def train(num_episodes=20000, ):
    """Train."""

    # Create the task environment.
    env = gym.make(name)

    # Create the DDPG agent in the task environment.
    agent = DDPG(env)

    with open(name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        i_step = 0
        for i_episode in range(1, num_episodes+1):
            # start a new episode
            state = agent.reset()
            sum_reward = 0.0
            N = 0
            while True:
                env.render()
                # Actor commands the action
                action = agent.act(state)
                # Environment reacts with next state, reward and done for
                # end-of-episode
                next_state, reward, done, info = env.step(action)
                # Agent (actor-critic) learns
                losses = agent.step(action, reward, next_state, done)
                # S <- S
                state = next_state
                sum_reward += reward
                N += 1
                i_step += 1
                if i_step % 1000 == 0:
                    loss_critic = losses
                    # End of episode. Show metrics.
                    to_write = (i_episode, i_step, loss_critic,
                                sum_reward/N)
                    print(
                        '\rEpisode: {:4d}, '
                        'Step: {:7d}, '
                        'Loss-crit: {:10.4f}, '
                        'Av Rwd: {:10.4f}, '
                        ''.format(*to_write)
                        )
                    # Re-use same line to print on.
                    # sys.stdout.flush()
                    # Write CSV row
                    for i, label in enumerate(labels):
                        results[label].append(to_write[i])
                    writer.writerow(to_write)
                if done:
                    break

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
