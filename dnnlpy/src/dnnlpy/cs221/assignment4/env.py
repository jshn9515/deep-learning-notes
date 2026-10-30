from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv

type RenderMode = Literal['human', 'rgb_array']

__all__ = ['CustomMountainCarEnv']


class CustomMountainCarEnv(MountainCarEnv):
    """A custom implementation of the MountainCar environment with adjustable parameters.

    This class extends the original MountainCarEnv from Gymnasium, allowing users to
    customize various parameters such as position limits, speed limits, goal position,
    and more. It also provides a method to register the custom environment in the Gym
    registry.

    For more details on the original MountainCar environment, refer to the :class:`MountainCarEnv`
    documentation in the Gymnasium library.

    Args:
        min_position (float, optional): Minimum position of the car. Defaults to -1.2.
        max_position (float, optional): Maximum position of the car. Defaults to 0.6.
        max_speed (float, optional): Maximum speed of the car. Defaults to 0.07.
        goal_position (float, optional): Position at which the episode is considered successful.
            Defaults to 0.5.
        goal_velocity (float, optional): Velocity at which the episode is considered successful.
            Defaults to 0.0.
        force (float, optional): The force applied to the car when an action is taken. Defaults
            to 0.001.
        gravity (float, optional): The gravitational constant affecting the car's movement.
            Defaults to 0.0025.
        render_mode (RenderMode, optional): The mode in which to render the environment. Can be
            'human' for rendering to the screen, 'rgb_array' for returning an RGB array, or
            ``None`` for no rendering. Defaults to ``None``.
    """

    def __init__(
        self,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.5,
        goal_velocity: float = 0.0,
        force: float = 0.001,
        gravity: float = 0.0025,
        *,
        render_mode: RenderMode | None = None,
    ):
        """Initializes the ``CustomMountainCarEnv`` with the specified parameters."""
        super().__init__(render_mode)
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.goal_velocity = goal_velocity
        self.force = force
        self.gravity = gravity

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(self.low, self.high)


def register_custom_mountain_car_env():
    """Registers the ``CustomMountainCar-v0`` environment in the Gym registry."""
    if 'CustomMountainCar-v0' not in gym.registry:

        def factory(**kwargs: Any) -> CustomMountainCarEnv:
            return CustomMountainCarEnv(**kwargs)

        gym.register(
            id='CustomMountainCar-v0',
            entry_point=factory,
            max_episode_steps=1000,
        )
