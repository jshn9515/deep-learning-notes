import random
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, override

import gymnasium as gym
import numpy as np

from . import utils

__all__ = [
    'MDP',
    'ContinuousGymMDP',
    'DiscreteGymMDP',
    'GymMDP',
    'MDPTransition',
    'NumberLineMDP',
]


class MDPTransition[StateT, ActionT](NamedTuple):
    """A single transition in an MDP.

    Attributes:
        state (StateT): The state before taking the action.
        action (ActionT): The action taken in the state.
        reward (float): The reward received after taking the action.
        next_state (StateT): The state after taking the action.
        terminal (bool): Whether the next state is terminal (i.e., the episode ended).
    """

    state: StateT
    action: ActionT
    reward: float
    next_state: StateT
    terminal: bool


class MDP[StateT, ActionT](ABC):
    """An abstract class representing a Markov Decision Process (MDP)."""

    @property
    @abstractmethod
    def actions(self) -> list[ActionT]:
        """Returns the list of possible actions at each state."""

    @property
    @abstractmethod
    def discount(self) -> float:
        r"""Returns the discount factor :math:`\gamma`."""

    @property
    @abstractmethod
    def time_limit(self) -> int:
        """Returns the maximum number of steps in an episode."""

    @property
    @abstractmethod
    def start_state(self) -> StateT:
        """Returns the start state of the MDP."""

    @abstractmethod
    def reset_state(self) -> None:
        """Resets the MDP to its initial state."""

    @abstractmethod
    def transition(self, action: ActionT) -> MDPTransition[StateT, ActionT]:
        """Takes action in the current state and returns :class:`MDPTransition`."""


class NumberLineMDP(MDP[int, int]):
    """A one-dimensional random walk Markov Decision Process.

    The agent starts at the center of a number line and moves left or right based on the
    selected action's transition probabilities. Each transition produces a reward, and
    the episode ends when either boundary is reached.

    Args:
        n (int, optional): The maximum absolute value of the states. The states will be
            in the range ``[-n, n]``. Defaults to 2.
        left_reward (float, optional): The reward for reaching the left terminal state
            ``-n``. Defaults to 10.
        right_reward (float, optional): The reward for reaching the right terminal state
            ``n``. Defaults to 50.
        penalty (float, optional): The penalty for reaching any non-terminal state.
            Defaults to -5.
        seed (int, optional): The seed for the random number generator. If ``None``, the
            random number generator will be initialized randomly. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int = 2,
        left_reward: float = 10,
        right_reward: float = 50,
        penalty: float = -5,
        seed: int | None = None,
    ):
        """Initialize the ``NumberLineMDP`` with the given parameters."""
        self.n = n
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.penalty = penalty
        self.rng = random.Random(seed)

        self.state = self.start_state
        self.terminal_states = {-n, n}
        self.indexer = utils.NumberLineIndexer(n)
        self.num_states = self.indexer.num_states

    @property
    @override
    def actions(self) -> list[int]:
        return [1, 2]  # 1: move right, 2: move left

    @property
    @override
    def discount(self) -> float:
        return 1.0

    @property
    @override
    def time_limit(self) -> int:
        return 100

    @property
    @override
    def start_state(self) -> int:
        return 0

    @property
    def all_states(self) -> list[int]:
        """Returns a list of all states in the MDP."""
        return self.indexer.all_states()

    @override
    def reset_state(self) -> None:
        self.state = 0

    @override
    def transition(self, action: int) -> MDPTransition[int, int]:
        """Takes action in the current state and returns :class:`MDPTransition`.

        Args:
            action (int): The action to take. Must be either 1 (move right) or 2 (move
                left).

        Returns:
            MDPTransition: A transition object containing the previous state, action,
                reward, next state, and terminal flag.
        """
        if self.state in self.terminal_states:
            raise RuntimeError('Attempting to call transition on a terminated MDP.')

        if action == 1:
            forward_prob = 0.2
        elif action == 2:
            forward_prob = 0.3
        else:
            raise ValueError('Invalid action provided.')

        prev_state = self.state
        if self.rng.random() < forward_prob:
            self.state += 1  # Move the agent forward
        else:
            self.state -= 1  # Move the agent backward

        if self.state == self.n:
            reward = self.right_reward
        elif self.state == -self.n:
            reward = self.left_reward
        else:
            reward = self.penalty

        # Check for termination
        terminal = self.state in self.terminal_states

        transition = MDPTransition(prev_state, action, reward, self.state, terminal)
        return transition

    def state_to_index(self, state: int) -> int:
        """Returns the index of the number line state."""
        return self.indexer.to_index(state)

    def index_to_state(self, index: int) -> int:
        """Returns the number line state with index."""
        return self.indexer.from_index(index)


class GymMDP(MDP[np.ndarray, int]):
    r"""Base class for MDPs wrapping Gymnasium environments with discrete action spaces.

    Provides a common interface for interacting with Gymnasium environments, including
    support for custom speed limits, discounting, and episode time limits.

    Args:
        env (str): Gymnasium environment ID, such as 'MountainCar-v0'.
        max_speed (float, optional): Maximum speed passed to the environment. Only
            supported by 'CustomMountainCar-v0'. Defaults to ``None``.
        discount (float, optional): Discount factor :math:`\gamma` for future rewards.
            Defaults to 0.99.
        time_limit (int, optional): Maximum number of steps per episode. If ``None``,
            uses the environment's default time limit.
        seed (int, optional): Seed for the random number generator. If ``None``, the
            random number generator will be initialized randomly. Defaults to ``None``.
    """

    def __init__(
        self,
        env: str,
        max_speed: float | None = None,
        discount: float = 0.99,
        time_limit: int | None = None,
        seed: int | None = None,
    ):
        """Initialize the ``GymMDP`` with the given parameters."""
        self.max_speed = max_speed

        if self.max_speed is not None:
            self.env = gym.make(env, max_speed=self.max_speed)
        else:
            self.env = gym.make(env)

        if not isinstance(self.env, gym.wrappers.TimeLimit):
            raise TypeError('GymMDP only supports environments wrapped in TimeLimit.')
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError(
                'GymMDP only supports environments with discrete action spaces.'
            )
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise TypeError(
                'GymMDP only supports environments with Box observation spaces.'
            )

        if time_limit is not None:
            self.env._max_episode_steps = time_limit

        self._actions = list(range(self.env.action_space.n))
        self._discount = discount
        self._time_limit = self.env._max_episode_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.state, _ = self.env.reset(seed=seed)
        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high

    @property
    @override
    def actions(self) -> list[int]:
        return self._actions.copy()

    @property
    @override
    def discount(self) -> float:
        return self._discount

    @property
    @override
    def time_limit(self) -> int:
        return self._time_limit

    @override
    def reset_state(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.state, _ = self.env.reset(seed=self.seed)

    def reward(self, next_state: np.ndarray, original_reward: float) -> float:
        """Compute the reward for transitioning to the given continuous state.

        For ``MountainCar`` environments, replaces the original reward with a shaped
        reward based on the car's position relative to the maximum position and the
        magnitude of its velocity. For other environments, the original reward is
        returned unchanged.

        Args:
            next_state (np.ndarray): Continuous state after the transition, containing
                the car's position and velocity.
            original_reward (float): Reward provided by the environment.

        Returns:
            float: Shaped reward for MountainCar environments, or the original reward for
                other environments.
        """
        if self.env.unwrapped.spec is None:
            raise AssertionError('Environment spec is not available.')

        if 'MountainCar-v0' in self.env.unwrapped.spec.id:
            # reward fn based on position and velocity
            position_reward = -(self.high[0] - next_state[0])
            velocity_reward = -(self.high[1] - np.abs(next_state[1]))
            return position_reward + velocity_reward
        else:
            return original_reward


class ContinuousGymMDP(GymMDP):
    """Gymnasium MDP whose state is the continuous observation array."""

    @property
    @override
    def start_state(self) -> np.ndarray:
        """Returns the start state of the MDP."""
        self.reset_state()
        return self.state

    @override
    def transition(self, action: int) -> MDPTransition[np.ndarray, int]:
        """Takes action in the current state and returns :class:`MDPTransition`."""
        state = self.state
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        reward = self.reward(next_state, float(reward))
        terminal = terminated or truncated
        return MDPTransition(state, action, reward, next_state, terminal)


class DiscreteGymMDP(GymMDP):
    """Gymnasium MDP with each observation dimension mapped to a bin.

    Args:
        env (str): Gymnasium environment ID.
        bins (int | list[int], optional): Bin edges per observation dimension.
        low (list[float], optional): Lower observation bounds.
        high (list[float], optional): Upper observation bounds.
        **kwargs: Additional :class:`GymMDP` arguments.
    """

    def __init__(
        self,
        env: str,
        bins: int | list[int] = 10,
        low: list[float] | None = None,
        high: list[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__(env, **kwargs)
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise TypeError(
                'DiscreteGymMDP only supports environments with Box observation spaces.'
            )
        if len(self.env.observation_space.shape) != 1:
            raise AssertionError(
                'DiscreteGymMDP only supports environments with 1D observation spaces.'
            )

        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

        # Convert the environment to a discretized version
        self.bins = utils.create_bins(self.low, self.high, bins)
        self.indexer = utils.BinnedStateIndexer(self.bins)
        self.num_states = self.indexer.num_states

    @property
    @override
    def start_state(self) -> np.ndarray:
        self.reset_state()
        return utils.discretize(self.state, self.bins)

    @property
    def all_states(self) -> list[int]:
        """Returns a list of all states in the MDP."""
        return self.indexer.all_states()

    @override
    def transition(self, action: int) -> MDPTransition[np.ndarray, int]:
        """Takes action in the current state and returns :class:`MDPTransition`."""
        state = utils.discretize(self.state, self.bins)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        reward = self.reward(next_state, float(reward))
        next_state = utils.discretize(next_state, self.bins)
        terminal = terminated or truncated
        return MDPTransition(state, action, reward, next_state, terminal)

    def state_to_index(self, state: np.ndarray) -> int:
        """Returns the index of the continuous state."""
        return self.indexer.to_index(state)

    def index_to_state(self, index: int) -> tuple[int, ...]:
        """Returns the discretized state with index."""
        return self.indexer.from_index(index)
