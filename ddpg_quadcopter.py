"""
DDPG agent

CREDITS
-------
- keras-rl : examples/ddpg_pendulum.py
- https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html - helping understand
  the lanuage of section 7 of the DDPG paper arXiv:1509.02971

"""
import sys

# Path to gym development version with QuadCopter-v0
sys.path.insert(0, '../gym/')
import gym

# Path to gym development version with QuadCopter-v0
sys.path.insert(0, '../keras-rl/')
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


ENV_NAME = 'QuadCopter-v0'
gym.undo_logger_setup()

HIDDEN_UNITS_1 = 300
HIDDEN_UNITS_2 = 600

NB_STEPS = 1000000
BATCH_SIZE = 64
LEARN_R = .0001
CLIPNORM = 1.

MEMORY = 1000000
WARMUP_ACTOR = 1
WARMUP_CRITIC = 1

THETA = 0.6
MU = 0.
SIGMA = 0.3

GAMMA=.99
TAU = 0.001

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
nb_observations = env.observation_space.shape[0]

# Map coef for x in [-1, 1] -> action in [low, high]
a = (env.action_space.high -  env.action_space.low) / 2.0
b = (env.action_space.high +  env.action_space.low) / 2.0
def action_map(x, a=None, b=None):
    z = a * x + b
    return z

# Actor
# init = RandomUniform(minval=-0.003, maxval=0.003)
init = RandomNormal(mean=0.0, stddev=0.003)
observation_input = Input((1, nb_observations,), name='A_observation_input')
flattened_observation = Flatten()(observation_input)
h0 = Dense(HIDDEN_UNITS_1, name='A_h0')(flattened_observation)
h0 = Activation('relu')(h0)
h1 = Dense(HIDDEN_UNITS_2, activation='relu', name='A_h1')(h0)
actions = Dense(nb_actions,  activation='tanh', name='A_last',
                kernel_initializer=init, bias_initializer=init)(h1)
actions = Lambda(action_map, arguments={'a': a, 'b': b}, name='A_map')(actions)
actor = Model(inputs=observation_input, outputs=actions)
print(actor.summary())


# Critic
action_input = Input((nb_actions,), name='Q_action_input')
observation_input = Input((1, nb_observations,), name='A_observation_input')
flattened_observation = Flatten()(observation_input)
s1 = Dense(HIDDEN_UNITS_1, activation='relu', name='Q_s1')(flattened_observation)
a1 = Dense(HIDDEN_UNITS_2, activation='linear', name='Q_a1')(action_input)
h1 = Dense(HIDDEN_UNITS_2, activation='linear', name='Q_h1')(s1)
h2 = Add(name='Q_h2')([h1,a1])
h3 = Dense(HIDDEN_UNITS_2, activation='relu', name='Q_h3')(h2)
Qvalues = Dense(1, activation='linear', name='Q_last')(h3)
# Qvalues = Flatten()(Qvalues)
critic = Model(inputs=[action_input, observation_input], outputs=Qvalues)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=MEMORY, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                          theta=THETA, mu=MU, sigma=SIGMA)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory,
                  batch_size=BATCH_SIZE,
                  nb_steps_warmup_actor=WARMUP_ACTOR,
                  nb_steps_warmup_critic=WARMUP_CRITIC,
                  random_process=random_process, gamma=GAMMA,
                  target_model_update=TAU)
agent.compile(Adam(lr=LEARN_R, clipnorm=CLIPNORM), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for
# show, but this slows down training quite a lot. You can always safely abort
# the training prematurely using Ctrl + C.
agent.fit(env, nb_steps=NB_STEPS,
          visualize=False,
          verbose=1,
          nb_max_episode_steps=1500)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=1500)
