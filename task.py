"""The Task (environment) module."""

import numpy as np
from physics_sim import PhysicsSim
from numpy import square as sqr


class Task():
    """Task (environment).

    Defines the goal and provides feedback to the agent.
    """

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])
        self.max_position_error = 20.0
        self.max_attitude_error = (22.0/7.0/180.0) * 30.0  # Thirty degrees

    def get_reward(self):
        """Use current pose of sim to return reward."""
        pos = self.sim.pose[0:3]
        d_pos = self.sim.v
        att = self.sim.pose[3:6]
        d_att = self.sim.angular_v

        pos_err = pos - self.target_pos
        att_err = att

        # Reward and also done and punish if exceeds maximum constraints.
        self.sim.done = self.sim.done  # Test if this works.
        if np.linalg.norm(pos_err) > self.max_position_error:
            self.sim.done = True
            reward = -10.0
        elif np.linalg.norm(att_err) > self.max_attitude_error:
            # Only valid for small angles phi and theta but psi is okay.
            self.sim.done = True
            reward = -10.0
        else:
            reward = 1.0 - (
                1.0 * sqr(pos_err)/sqr(self.max_position_error) +
                1.0 * sqr(att)/sqr(self.max_attitude_error) +
                0.008 * sqr(d_pos)/sqr(self.max_position_error) +
                0.008 * sqr(d_att)/sqr(self.max_attitude_error)
            ).sum()

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
