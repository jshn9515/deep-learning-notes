from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def tiny_features() -> Callable[[np.ndarray], np.ndarray]:
    """The tests use this tiny feature extractor instead of yours, so a bug in 4b does
    not affect them. With 4 features and 3 actions, theta has shape ``(4, 3)``.
    """
    return lambda s: np.array([1.0, s[0], s[1], s[0] * s[1]])


@pytest.fixture
def initial_theta() -> np.ndarray:
    """A small theta for testing. With 4 features and 3 actions, theta has shape ``(4, 3)``."""
    return np.array(
        [
            [0.1, -0.2, 0.3],
            [0.5, 0.0, -0.5],
            [-1.0, 2.0, 0.0],
            [0.4, -0.3, 0.2],
        ]
    )


@pytest.fixture
def bootstrap_states() -> list[tuple[float, float]]:
    return [(0.5, 0.2), (-1.0, 0.5), (0.0, 0.0), (0.3, -0.4)]


@pytest.fixture
def bootstrap_actions() -> list[int]:
    return [2, 0, 1]


@pytest.fixture
def bootstrap_rewards():
    return [-1.0, -0.5, 2.0]
