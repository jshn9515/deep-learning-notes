from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .algorithms import RLAlgorithm
    from .mdp import MDP

__all__ = [
    'BinnedStateIndexer',
    'NumberLineIndexer',
    'fourier_feature_extractor',
    'polynomial_feature_extractor',
    'simulate',
]

logger = logging.getLogger(__name__)


class Indexer[StateT](ABC):
    """An abstract base class for mapping states to contiguous indices."""

    @abstractmethod
    def to_index(self, state: StateT) -> int:
        """Returns the index of the given state."""

    @abstractmethod
    def from_index(self, index: int) -> StateT:
        """Returns the state with the given index (the inverse of :meth:`to_index`)."""

    @abstractmethod
    def all_states(self) -> list[StateT]:
        """Returns all the states in order of their indices."""


class NumberLineIndexer(Indexer[int]):
    r"""Utility for mapping integer number line states to contiguous indices.

    Args:
        n (int): The maximum absolute value of the states. The states will be in the range
            :math:`[-n, n]`.
    """

    def __init__(self, n: int) -> None:
        r"""Initialize the indexer for states :math:`\{-n,\dots,n\}`."""
        self.n = n
        self.num_states = 2 * self.n + 1

    @override
    def to_index(self, state: int) -> int:
        """Returns the index of the number line state (:math:`-n` has index 0)."""
        return state + self.n

    @override
    def from_index(self, index: int) -> int:
        """Returns the number line state with index (the inverse of :meth:`to_index`)."""
        return index - self.n

    @override
    def all_states(self) -> list[int]:
        r"""Returns all the states :math:`\{-n,\dots,n\}`, in order of their indices."""
        return list(range(-self.n, self.n + 1))


def create_bins(
    low: list[float],
    high: list[float],
    bins: int | list[int],
) -> list[np.ndarray]:
    """Takes in the lower and upper bounds of each feature and returns a list with one array
    of :attr:`bins` evenly spaced bin edges per feature.

    Args:
        low (list[float]): The lower bounds of each feature.
        high (list[float]): The upper bounds of each feature.
        bins (int | list[int]): The number of bins for each feature. If an integer is
            provided, the same number of bins will be used for all features. If a list is
            provided, it must have the same length as :attr:`low` and :attr:`high`, and
            each element specifies the number of bins for the corresponding feature.

    Returns:
        list[np.ndarray]: A list of arrays, where each array contains the bin edges for
            a feature.

    Raises:
        AssertionError: If :attr:`low` and :attr:`high` do not have the same length
        AssertionError: If :attr:`num_bins` is a list and does not have the same length as
            :attr:`low` and :attr:`high`.
    """
    if len(low) != len(high):
        raise AssertionError('`low` and `high` must have the same length.')
    if isinstance(bins, list) and len(bins) != len(low):
        raise AssertionError('`num_bins` must have same length as `low` and `high`.')

    if isinstance(bins, int):
        bins = [bins for _ in range(len(low))]

    bins_arr: list[np.ndarray] = []
    for l, h, n in zip(low, high, bins, strict=True):
        bins_arr.append(np.linspace(l, h, n))

    return bins_arr


def discretize(x: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    """Discretizes an array :attr:`x` according to the provided :attr:`bins`.

    Args:
        x (np.ndarray): The continuous observation to be discretized.
        bins (list[np.ndarray]): A list of bin edge arrays, one per feature.

    Returns:
        np.ndarray: An array of bin indices corresponding to the discretized observation.
    """
    multi_index: list[int] = []
    for feature, bin_edges in zip(x, bins, strict=True):
        index = int(np.digitize(feature, bin_edges))
        multi_index.append(index)
    return np.array(multi_index)


class BinnedStateIndexer(Indexer[np.ndarray]):
    """Utility for mapping continuous states to discrete states using bins.

    Args:
        bins (list[np.ndarray]): A list of bin edges for each dimension of the state space.
    """

    def __init__(self, bins: list[np.ndarray]):
        """Initialize the indexer with the given bins."""
        self.bins = bins
        self.state_shapes = tuple(len(b) + 1 for b in bins)
        self.num_states = math.prod(self.state_shapes)

    @override
    def to_index(self, state: np.ndarray) -> int:
        """Returns the flat index of an already discretized state."""
        index = np.ravel_multi_index(state, self.state_shapes)
        return int(index)

    @override
    def from_index(self, index: int) -> tuple[int, ...]:
        """Returns the tuple of bin indices with state index (the inverse of :meth:`to_index`)."""
        multi_index = np.unravel_index(index, self.state_shapes)
        return tuple(int(i) for i in multi_index)

    @override
    def all_states(self) -> list[int]:
        r"""Returns all the state indices :math:`\{0,\dots,n-1\}`."""
        return list(range(self.num_states))


def polynomial_feature_extractor(
    state: np.ndarray,
    degree: int = 3,
    scale: npt.ArrayLike | None = None,
) -> np.ndarray:
    r"""Extracts polynomial features from the given state up to the specified degree.

    For a state with :math:`d` dimensions, this function computes all polynomial combinations
    of the state dimensions up to the specified degree, including interaction terms. The resulting
    feature vector will have a length of :math:`\binom{d + \text{degree}}{\text{degree}}`, which
    includes all combinations of the state dimensions up to the specified degree.

    For example, we have a 3D state :math:`(x, y, z)` with degree 2, and scale :math:`[2, 1, 1]`,
    the output will include features like:

    .. math::

        \[1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, \dots, 4x^2y^2z^2\]

    Args:
        state (np.ndarray): The input state, a NumPy array of shape ``(features,)``.
        degree (int, optional): The maximum degree of the polynomial features. Defaults to 3.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the polynomial features. If not
            provided, no scaling is applied. Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the polynomial features of the input state, including
            all combinations of the state dimensions up to the specified degree.

    Examples:
        >>> state = np.array([1, 2])
        >>> features = polynomial_feature_extractor(state, degree=2)
    """
    if scale is None:
        scale = np.ones_like(state)
    else:
        scale = np.asarray(scale)

    features = np.array([1.0])
    for s, sc in zip(state, scale, strict=True):
        features = np.outer(features, (s * sc) ** np.arange(degree + 1))
        features = features.flatten()

    return features


# Problem 4b: Fourier feature extractor
def fourier_feature_extractor(
    state: np.ndarray,
    max_coeff: int = 5,
    scale: npt.ArrayLike | None = None,
) -> np.ndarray:
    r"""Extracts Fourier features from the given state up to the specified maximum coefficient.

    For a state with :math:`d` dimensions, this function computes the Fourier features of the
    state dimensions up to the specified maximum coefficient. The resulting feature vector will
    have a length of :math:`(max\_coeff + 1)^d`, which includes all combinations of the state
    dimensions up to the specified maximum coefficient.

    .. math::

        \[1, \cos(\pi c_1 s_1), \cos(\pi c_2 s_2), \ldots, \cos(\pi c_d s_d),
        \cos(\pi c_1 s_1 + \pi c_2 s_2 + \ldots + \pi c_d s_d), \ldots\]

    Args:
        state (np.ndarray): The input state, a NumPy array of shape ``(features,)``.
        max_coeff (int, optional): The maximum coefficient for the Fourier features.
            Defaults to 5.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the Fourier features. If not
            provided, no scaling is applied. Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the Fourier features of the input state, including
            all combinations of the state dimensions up to the specified maximum coefficient.

    Examples:
        >>> state = np.array([0.5, 1.0])
        >>> features = fourier_feature_extractor(state, max_coeff=2)
    """
    if scale is None:
        scale = np.ones_like(state)
    else:
        scale = np.asarray(scale)

    features = np.array([0.0])
    for s, sc in zip(state, scale, strict=True):
        new_features = np.arange(max_coeff + 1) * (s * sc)
        features = features[:, None] + new_features[None, :]
        features = features.flatten()

    return np.cos(np.pi * features)


def _render_mdp[StateT, ActionT](mdp: MDP[StateT, ActionT]):
    """Renders the MDP environment if it is a GymMDP."""
    from .mdp import GymMDP

    if not isinstance(mdp, GymMDP):
        raise TypeError('Demo mode is only supported for GymMDP.')

    mdp.env.render()


def simulate[StateT, ActionT](
    mdp: MDP[StateT, ActionT],
    rl: RLAlgorithm[StateT, ActionT],
    num_episodes: int = 100,
    train: bool = True,
    verbose: bool = True,
    demo: bool = False,
) -> np.ndarray:
    """Runs a simulation of the given MDP and RL algorithm for a specified number of trials.

    On each trial, the RL algorithm interacts with the MDP according to its dynamics, and
    the discounted reward is calculated for each trial. A trial ends when a terminal state
    is reached, after the MDP's time limit is reached, or when the RL algorithm returns
    ``None`` for an action (problem 5c).

    Args:
        mdp (MDP): The Markov Decision Process to simulate.
        rl (RLAlgorithm): The reinforcement learning algorithm to be simulated.
        num_episodes (int, optional): The number of episodes to run. Defaults to 100.
        train (bool, optional): Whether to allow the RL algorithm to learn from feedback.
            Defaults to ``True``.
        verbose (bool, optional): Whether to print progress information during the simulation.
            Defaults to ``True``.
        demo (bool, optional): Whether to render the MDP environment during the simulation.
            Defaults to ``False``.

    Returns:
        np.ndarray: A 1D array containing the discounted rewards obtained from each trial.
    """
    total_rewards: list[float] = []  # The discounted rewards we get on each trial

    for episode in range(num_episodes):
        state = mdp.start_state

        if demo:
            _render_mdp(mdp)

        total_discount = 1
        total_reward = 0
        episode_length = 0

        for _ in range(mdp.time_limit):
            if demo:
                time.sleep(0.05)

            action = rl.get_action(state, explore=train)
            transition = mdp.transition(action)
            episode_length += 1

            if train:
                rl.step(transition)

            total_reward += total_discount * transition.reward
            total_discount *= mdp.discount
            state = transition.next_state

            if transition.terminal:
                break  # We have reached a terminal state

        if verbose and episode % 100 == 0:
            n = len(str(num_episodes))
            logger.info(
                f'Episode {episode:{n}} | reward: {total_reward:.4f}, '
                f'length: {episode_length:d}'
            )

        total_rewards.append(total_reward)

    return np.array(total_rewards)
