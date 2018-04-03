"""Main run."""

import getopt
from glob import glob
import os
import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task
import numpy as np


def train(num_episodes=1000, ):
    """Train."""
    target_pos = np.array([0., 0., 10.])
    task = Task(target_pos=target_pos)
    agent = PolicySearch_Agent(task)

    for i_episode in range(1, num_episodes+1):
        # start a new episode
        state = agent.reset_episode()
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(reward, done)
            state = next_state
            if done:
                print('\rEpisode = {:4d}, score = {:7.3f} '
                      '(best = {:7.3f}), noise_scale = {}'.format(
                          i_episode,
                          agent.score, agent.best_score,
                          agent.noise_scale), end='')
                break
        sys.stdout.flush()


def main(argv):
    """Command line parser for batch processes."""
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
