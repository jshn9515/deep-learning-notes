import math

import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4
import dnnlpy.cs221.assignment4.utils as utils


def test_number_line_indexer_round_trip():
    indexer = utils.NumberLineIndexer(2)

    assert indexer.num_states == 5
    assert indexer.all_states() == [-2, -1, 0, 1, 2]
    assert [indexer.to_index(s) for s in indexer.all_states()] == list(range(5))
    assert [indexer.from_index(i) for i in range(5)] == indexer.all_states()


def test_create_bins_supports_different_edge_counts():
    bins = utils.create_bins([-1.0, 0.0], [1.0, 6.0], [3, 4])

    npt.assert_allclose(bins[0], [-1.0, 0.0, 1.0])
    npt.assert_allclose(bins[1], [0.0, 2.0, 4.0, 6.0])


def test_create_bins_uses_same_edge_count_for_all_dimensions():
    bins = utils.create_bins([-1.0, 0.0], [1.0, 6.0], 3)

    npt.assert_allclose(bins[0], [-1.0, 0.0, 1.0])
    npt.assert_allclose(bins[1], [0.0, 3.0, 6.0])


@pytest.mark.parametrize(
    ('observation', 'expected'),
    [
        ([-2.0, -1.0], [0, 0]),
        ([-1.0, 0.0], [1, 1]),
        ([0.5, 3.0], [2, 2]),
        ([1.0, 6.0], [3, 4]),
        ([2.0, 7.0], [3, 4]),
    ],
)
def test_discretize_handles_edges_and_out_of_range_values(
    observation: list[float], expected: list[int]
):
    x = np.array(observation)
    bins = utils.create_bins([-1.0, 0.0], [1.0, 6.0], [3, 4])

    actual = utils.discretize(x, bins)
    npt.assert_equal(actual, expected)


def test_binned_state_indexer_is_one_to_one_for_all_bin_pairs():
    bins = utils.create_bins([-1.0, 0.0], [1.0, 6.0], [2, 3])
    indexer = utils.BinnedStateIndexer(bins)
    indices = [
        indexer.to_index(np.array([position, velocity]))
        for position in range(3)
        for velocity in range(4)
    ]

    assert indexer.num_states == 12
    assert indices == indexer.all_states()

    for index in indices:
        state = np.array(indexer.from_index(index))
        assert indexer.to_index(state) == index


def test_binned_state_indexer_flattens_existing_bin_numbers():
    bins = utils.create_bins([-1.2, -0.07], [0.6, 0.07], 20)
    indexer = utils.BinnedStateIndexer(bins)

    assert indexer.to_index(np.array([8, 11])) == 8 * 21 + 11
    assert indexer.from_index(8 * 21 + 11) == (8, 11)


@pytest.mark.parametrize('state', [[-1, 0], [0, 4]])
def test_binned_state_indexer_rejects_invalid_bin_numbers(state: list[int]):
    bins = utils.create_bins([-1.0, 0.0], [1.0, 6.0], [2, 3])
    indexer = utils.BinnedStateIndexer(bins)

    with pytest.raises(ValueError):
        indexer.to_index(np.array(state))


def test_polynomial_feature_extractor_with_scaling():
    actual = utils.polynomial_feature_extractor(
        np.array([2.0, 3.0]), degree=1, scale=[0.5, 2.0]
    )
    expected = [1.0, 6.0, 1.0, 6.0]

    npt.assert_allclose(actual, expected)


def test_fourier_feature_extractor_for_two_features():
    actual = utils.fourier_feature_extractor(np.array([0.5, 0.25]), max_coeff=1)
    expected = [1.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5)]

    npt.assert_allclose(actual, expected, atol=1e-15)


class TwoStepMDP(a4.MDP[int, int]):
    state = 0

    @property
    def actions(self) -> list[int]:
        return [1]

    @property
    def discount(self) -> float:
        return 0.5

    @property
    def time_limit(self) -> int:
        return 3

    @property
    def start_state(self) -> int:
        self.reset_state()
        return self.state

    def reset_state(self) -> None:
        self.state = 0

    def transition(self, action: int) -> a4.MDPTransition[int, int]:
        state = self.state
        self.state += 1

        return a4.MDPTransition(
            state, action, float(self.state), self.state, self.state == 2
        )


def test_simulate_stops_on_terminal_and_discounts_rewards():
    mdp = TwoStepMDP()
    rl = a4.FixedRLAlgorithm({0: 1, 1: 1}, mdp.actions, epsilon=0)

    actual = utils.simulate(mdp, rl, num_episodes=2, train=False, verbose=False)
    expected = [2.0, 2.0]
    npt.assert_allclose(actual, expected)
