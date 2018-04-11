"""
DDPG agent
----------

CREDITS
-------
- Deep Deterministic Policy Gradients (DDPG)
  https://arxiv.org/pdf/1509.02971.pdf

- keras-rl : examples/ddpg_pendulum.py

- Ben Lau : https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html for
  helping understand the lanuage of section 7 of the DDPG paper arXiv:1509.02971

- kkweon : https://gist.github.com/kkweon/a82980f3d60ffce1d69ad6da8af0e124
  for helping with the ArgumentParser

"""
import sys

# Path to gym development version with QuadCopter-v0
sys.path.insert(0, './gym/')
import gym

# Path to gym development version with QuadCopter-v0
sys.path.insert(0, './keras-rl/')
import rl

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Concatenate
from keras.layers import Flatten, Reshape
from keras.layers import Add
from keras.layers import Dropout, BatchNormalization
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger
import argparse

ENV_NAME = 'QuadCopter-v0'
gym.undo_logger_setup()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--test_episodes",
                    type=int ,
                    dest="TEST_EPISODES",
                    default=5 ,
                    help="Number of testing episodes")
parser.add_argument("--hidden_units_1",
                    type=int,
                    dest="HIDDEN_UNITS_1",
                    default=300,
                    help="Number of units in hidden later 1")
parser.add_argument("--hidden_units_2",
                    type=int,
                    dest="HIDDEN_UNITS_2",
                    default=600,
                    help="Number of units in hidden later 2")
parser.add_argument("--nb_steps",
                    type=int,
                    dest="NB_STEPS",
                    default=1000000,
                    help="Training steps")
parser.add_argument("--batch_size",
                    type=int,
                    dest="BATCH_SIZE",
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--learn_r",
                    type=float,
                    dest="LEARN_R",
                    default=.0001,
                    help="Learning rate")
parser.add_argument("--dropout",
                    type=float,
                    dest="DROPOUT",
                    default=0.3,
                    help="Dropout rate")
parser.add_argument("--clipnorm",
                    type=float,
                    dest="CLIPNORM",
                    default=1.0,
                    help="Gradient clipping value (positive)")
parser.add_argument("--memory",
                    type=int,
                    dest="MEMORY",
                    default=1000000,
                    help="Capacity of the ReplayMemory")
parser.add_argument("--warmup-actor",
                    type=int,
                    dest="WARMUP_ACTOR",
                    default=500,
                    help="Number of steps before training Actor")
parser.add_argument("--warmup-critic",
                    type=int,
                    dest="WARMUP_CRITIC",
                    default=500,
                    help="Number of steps before training Actor")
parser.add_argument("--action-init-var",
                    type=float,
                    dest="ACTION_INIT_VAR",
                    default=0.0001,
                    help="Actor output layer initializer variance")
parser.add_argument("--theta",
                    type=float,
                    dest="THETA",
                    default=0.6,
                    help="Ornstein-Uhlenbeck noise mean reversion rate")
parser.add_argument("--mu",
                    type=float,
                    dest="MU",
                    default=0.0,
                    help="Ornstein-Uhlenbeck noise mean")
parser.add_argument("--sigma",
                    type=float,
                    dest="SIGMA",
                    default=0.3,
                    help="Ornstein-Uhlenbeck noise variance")
parser.add_argument("--gamma",
                    type=float,
                    dest="GAMMA",
                    default=0.99,
                    help="Gamma, the discount rate")
parser.add_argument("--tau",
                    type=float,
                    dest="TAU",
                    default=0.001,
                    help="Tau for soft update (the lower the softer update)")
parser.add_argument("--verbose",
                    type=int,
                    dest="VERBOSE",
                    default=0,
                    help="Verbosity of training output")
parser.add_argument("--log-file",
                    type=str,
                    dest="LOG_FILE",
                    default="data.json",
                    help="File for logging metrics to")
# parser.add_argument("--log-interval",
#                     type=int,
#                     dest="LOG_INTERVAL",
#                     default=1000,
#                     help="Interval for file logging of metrics")
HYP = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
nb_observations = env.observation_space.shape[0]

#  Log data to file
if HYP.VERBOSE == 3:
    # FloydHub
    log_filename = '/output/' + HYP.LOG_FILE
else:
    # Local machine
    log_filename = HYP.LOG_FILE
file_logger = FileLogger(log_filename)

# Actor
# init = RandomUniform(minval=-0.003, maxval=0.003)
init = RandomNormal(mean=0.0, stddev=HYP.ACTION_INIT_VAR)
observation_input = Input((1, nb_observations,), name='A_observation_input')
flattened_observation = Flatten()(observation_input)

h0 = Dense(HYP.HIDDEN_UNITS_1, name='A_h0')(flattened_observation)
h0 = Dropout(HYP.DROPOUT)(h0)
h0 = Activation('relu')(h0)

h1 = Dense(HYP.HIDDEN_UNITS_2, name='A_h1')(h0)
h1 = Dropout(HYP.DROPOUT)(h1)
h1 = Activation('relu')(h1)

h2 = Dense(HYP.HIDDEN_UNITS_2, name='A_h2')(h1)
h2 = Dropout(HYP.DROPOUT)(h2)
h2 = Activation('relu')(h2)

h3 = Dense(HYP.HIDDEN_UNITS_2, name='A_h3')(h2)
h3 = Dropout(HYP.DROPOUT)(h3)
h3 = Activation('relu')(h3)

actions = Dense(nb_actions, name='A_last',
                kernel_initializer=init, bias_initializer=init)(h3)
actions = Dropout(HYP.DROPOUT)(actions)
actions = Activation('tanh')(actions)

actor = Model(inputs=observation_input, outputs=actions)
print(actor.summary())


# Critic
action_input = Input((nb_actions,), name='Q_action_input')
observation_input = Input((1, nb_observations,), name='A_observation_input')
flattened_observation = Flatten()(observation_input)

s1 = Dense(HYP.HIDDEN_UNITS_1, name='Q_s1')(flattened_observation)
s1 = Dropout(HYP.DROPOUT)(s1)
s1 = Activation('relu')(s1)

a1 = Dense(HYP.HIDDEN_UNITS_2, name='Q_a1')(action_input)
a1 = Dropout(HYP.DROPOUT)(a1)
a1 = Activation('linear')(a1)

h1 = Dense(HYP.HIDDEN_UNITS_2, name='Q_h1')(s1)
h1 = Dropout(HYP.DROPOUT)(h1)
h1 = Activation('linear')(h1)

h_add = Add(name='Q_add')([h1,a1])

h2 = Dense(HYP.HIDDEN_UNITS_2, name='Q_h2')(h_add)
h2 = Dropout(HYP.DROPOUT)(h2)
h2 = Activation('relu')(h2)

h3 = Dense(HYP.HIDDEN_UNITS_2, name='Q_h3')(h2)
h3 = Dropout(HYP.DROPOUT)(h3)
h3 = Activation('relu')(h3)

h4 = Dense(HYP.HIDDEN_UNITS_2, name='Q_h4')(h3)
h4 = Dropout(HYP.DROPOUT)(h4)
h4 = Activation('relu')(h4)

Qvalues = Dense(1, activation='linear', name='Q_last')(h4)

critic = Model(inputs=[action_input, observation_input], outputs=Qvalues)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=HYP.MEMORY, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                          theta=HYP.THETA,
                                          mu=HYP.MU,
                                          sigma=HYP.SIGMA)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory,
                  batch_size=HYP.BATCH_SIZE,
                  nb_steps_warmup_actor=HYP.WARMUP_ACTOR,
                  nb_steps_warmup_critic=HYP.WARMUP_CRITIC,
                  random_process=random_process, gamma=HYP.GAMMA,
                  target_model_update=HYP.TAU)
agent.compile(Adam(lr=HYP.LEARN_R, clipnorm=HYP.CLIPNORM), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for
# show, but this slows down training quite a lot. You can always safely abort
# the training prematurely using Ctrl + C.
agent.fit(env, nb_steps=HYP.NB_STEPS,
          visualize=False,
          callbacks=[file_logger],
          verbose=HYP.VERBOSE)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=HYP.TEST_EPISODES,
           visualize=False,
           nb_max_episode_steps=1500)
