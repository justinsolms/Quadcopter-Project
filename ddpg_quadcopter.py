"""
DDPG agent

CREDITS
-------
keras-rl : examples/ddpg_pendulum.py

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
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.layers import Dropout, BatchNormalization
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.initializers import RandomUniform

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


ENV_NAME = 'QuadCopter-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]


# Map coef for x in [-1, 1] -> action in [low, high]
a = (env.action_space.high -  env.action_space.low) / 2.0
b = (env.action_space.high +  env.action_space.low) / 2.0
def action_map(x, a=None, b=None):
    z = a * x + b
    return z

# Next, we build a very simple model.
init_w = RandomUniform(minval=-0.003, maxval=0.003)
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input_1')
y = Flatten(input_shape=(1,) + env.observation_space.shape)(observation_input)
y = Dense(1024)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(256)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(128)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(nb_actions,
                kernel_initializer=init_w,
                bias_initializer='zeros')(y)
y = Activation('tanh')(y)
y = Dropout(0.5)(y)
y = Lambda(action_map, arguments={'a': a, 'b': b})(y)
actor = Model(inputs=[observation_input], outputs=y)
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input_2')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
x = Activation('linear')(x)
x = Dropout(0.5)(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                          theta=0.15, mu=0., sigma=0.4)  # 0.3
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory,
                  nb_steps_warmup_critic=50000,
                  nb_steps_warmup_actor=50000,
                  random_process=random_process, gamma=.99,
                  target_model_update=0.001)
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for
# show, but this slows down training quite a lot. You can always safely abort
# the training prematurely using Ctrl + C.
agent.fit(env, nb_steps=1000000,
          visualize=False,
          verbose=1,
          nb_max_episode_steps=1500)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=1500)
