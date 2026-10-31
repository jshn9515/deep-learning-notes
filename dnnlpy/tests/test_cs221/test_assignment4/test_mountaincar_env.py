import gymnasium as gym
import numpy as np
import numpy.testing as npt
import pytest
from gymnasium.envs.classic_control import MountainCarEnv
from gymnasium.utils.env_checker import check_env

import dnnlpy.cs221.assignment4 as a4
from dnnlpy.cs221.assignment4.utils import discretize


def test_custom_mountain_car_env():
    env = gym.make('CustomMountainCar-v0')
    check_env(env.unwrapped)


@pytest.mark.parametrize('mdp_type', [a4.ContinuousGymMDP, a4.DiscreteGymMDP])
def test_mountain_car_transition_tracks_current_state(mdp_type: type[a4.GymMDP]):
    mdp = mdp_type('MountainCar-v0', seed=0)
    assert isinstance(mdp.env.unwrapped, MountainCarEnv)

    first = mdp.transition(2)
    npt.assert_allclose(mdp.state, mdp.env.unwrapped.state, atol=1e-7)

    second = mdp.transition(2)
    npt.assert_equal(second.state, first.next_state)
    npt.assert_allclose(mdp.state, mdp.env.unwrapped.state, atol=1e-7)


def test_discrete_mountain_car_state_indices_preserve_bins():
    mdp = a4.DiscreteGymMDP('MountainCar-v0', bins=20, seed=0)
    left = np.array([1, 1])
    right = np.array([2, 1])

    assert mdp.state_to_index(left) != mdp.state_to_index(right)
    assert mdp.index_to_state(mdp.state_to_index(left)) == (1, 1)
    assert mdp.index_to_state(mdp.state_to_index(right)) == (2, 1)


def test_discrete_mountain_car_tracks_states_across_multiple_bins():
    mdp = a4.DiscreteGymMDP('MountainCar-v0', bins=20, seed=0)
    state = mdp.start_state
    start_state = state.copy()
    visited = {mdp.state_to_index(state)}

    for _ in range(100):
        transition = mdp.transition(2)
        npt.assert_equal(transition.state, state)
        npt.assert_equal(transition.next_state, discretize(mdp.state, mdp.bins))
        state = transition.next_state
        visited.add(mdp.state_to_index(state))

    assert len(visited) > 1
    npt.assert_equal(mdp.start_state, start_state)
