"""The Task (environment) module."""

import numpy as np
from physics_sim import PhysicsSim
import gym
from numpy import square as sqr
from numpy.linalg import norm


class Task():
    """Task (environment).

    Defines the goal and provides feedback to the agent.
    """

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5.0, target_pos=None):
        """Initialize a Task object.

        Parameters
        ----------
        init_pose:
            initial position of the quadcopter in (x,y,z) dimensions and the
            Euler angles
        init_velocities:
            initial velocity of the quadcopter in (x,y,z) dimensions
        init_angle_velocities:
            initial radians/second for each of the three Euler angles
        runtime:
            time limit for each episode
        target_pos:
            target/goal (x,y,z) position for the agent

        State Attributes
        ----------------
        sim.pose:
            The position of the quadcopter in x,y,z dimensions and the
            Euler angles.
        sim.v:
            The velocity of the quadcopter in x,y,z dimensions.
        sim.angular_v:
            Angular velocity in radians/second for each Euler angle.


        """
        # Simulation
        self.sim = make('Pendulum-v0')
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 405 - 20
        self.action_high = 405 + 20
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])
        self.max_position_error = 10.0
        self.max_attitude_error = (22.0/7.0/180.0) * 15.0  # Thirty degrees

    def get_reward(self):
        """Use current pose of sim to return reward."""
        pos = self.sim.pose[0:3]
        vel = self.sim.v
        att = self.sim.pose[3:6]
        rot = self.sim.angular_v

        e_pos = (self.target_pos - pos) / self.max_position_error
        e_att = - np.sin(att) / np.sin(self.max_attitude_error)

        # Reward and also done and punish if exceeds maximum constraints.
        if np.any(np.abs(e_att) > 1.0):
            # Only valid for small angles phi and theta but psi is okay.
            self.sim.done = True
            reward = -10.0
        else:
            reward = (
                + 0.5
                - 0.001 * np.dot(e_pos, e_pos)
                + 0.001 * np.dot(e_pos, vel)
                - 1.0 * np.dot(e_att, e_att)
                ).sum()
        # print(e_pos, vel, e_att, reward)
        return reward

    def step(self, rotor_speeds):
        """Use action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
