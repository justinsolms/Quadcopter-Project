"""Deep Deterministic Policy Gradients (DDPG) implementation.

CREDITS
-------
    Models copied from kkweon/DDPG.py
    https://gist.github.com/kkweon/a82980f3d60ffce1d69ad6da8af0e124

    Basic layout from Udacity rl-quad-copter project.

"""

from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
from utils import OUNoise, ReplayBuffer
import numpy as np


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learning_rate = 1.0e-4

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        net = layers.Dense(16)(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        net = layers.Dense(16)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        net = layers.Dense(16)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Try different layer sizes, activations, add batch normalization,
        # regularizers, etc.

        # Add final output layer with sigmoid activations for the actions vector
        # elements in the range [0, 1]
        init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        raw_actions = layers.Dense(units=self.action_size,
                                   activation='sigmoid',
                                   kernel_initializer=init,
                                   bias_initializer=init,
                                   name='raw_actions',
                                   )(net)

        # Scale [0, 1] output for each action dimension to proper copter rotor
        # command range
        actions = layers.Lambda(
            lambda x: (x * self.action_range) + self.action_low, name='actions'
        )(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],  # NOTE: Hoping this outputs loss.
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.learning_rate = 1.0e-3

        self.build_model()

    def build_model(self):
        """Build a critic.

        This is a (value) network that maps (state, action) pairs -> Q-values.
        """
        # Initializers.
        init = initializers.VarianceScaling(scale=1.0 / 3.0,
                                            mode='fan_in',
                                            distribution='uniform')

        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(32)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(32)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization,
        # regularizers, etc.

        # Combine state and action pathways
        net = layers.Concatenate()([net_states, net_actions])

        # Add more layers to the combined network if needed
        net = layers.Dense(32)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final linear output layer to prduce action values (Q values)
        init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        Q_values = layers.Dense(units=1,
                                kernel_initializer=init,
                                bias_initializer=init,
                                name='q_values',
                                kernel_regularizer=regularizers.l2(0.01)
                                )(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss
        # function
        optimizer = optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by
        # actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, env):
        """Class initialization."""
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.high[0]
        self.action_high = env.action_space.low[0]

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size,
                                 self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(
            self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(
            self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu,
                             self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset(self):
        """Start a new episode."""
        self.noise.reset()
        state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        """Save in experience buffer and batch learn from buffer step.

        Save the action, reward, next_state in the experience buffer and if the
        buffer has enough samples to satisfy the batch size then make a learning
        step.

        """
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # if len(self.memory) > self.batch_size:
        if len(self.memory) > self.batch_size * 50:
            experiences = self.memory.sample()
            loss_critic = self.learn(experiences)
        else:
            loss_critic = None

        # Roll over last state and action
        self.last_state = next_state

        return loss_critic

    def act(self, state):
        """Return actions for given state(s) as per current policy.

        Also add some noise to the action (control-command) to explore the
        space.

        """
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        # add some noise for exploration
        return list(action + self.noise.sample())

    def learn(self, experiences):
        """Update policy and value parameters.

        Use given batch of experience tuples from the experience buffer.

        """

        # Convert experience tuples to separate arrays for each element (states,
        # actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]
                           ).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]
                           ).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]
                         ).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack(
            [e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from (target) models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch(
            [next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        loss_critic = self.critic_local.model.train_on_batch(
            x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(
            [states, actions, 0]), (-1, self.action_size))
        # Customized actor training function
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        return loss_critic

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.

        Update the target model with the local model weights. Do so gradually
        by using a soft update parameter, tau.

        Note
        ----
        After training over a batch of experiences, we could just copy our newly
        learned weights (from the local model) to the target model. However,
        individual batches can introduce a lot of variance into the process, so
        it's better to perform a soft update, controlled by the parameter tau.

        """
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), \
            "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + \
            (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
