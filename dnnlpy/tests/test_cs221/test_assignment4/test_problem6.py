import math
from collections.abc import Callable

import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4

from ..utils import timeout

type FeatureExtractor = Callable[[np.ndarray], np.ndarray]


def make_policy_gradient(
    feature_extractor: FeatureExtractor,
    discount: float = 0.9,
    time_limit: int = 1000,
    step_size: float = 0.1,
    baseline_step_size: float = 0.25,
) -> a4.PolicyGradient:
    """Create a PolicyGradient instance with the given parameters."""
    return a4.PolicyGradient(
        feature_dim=4,
        feature_extractor=feature_extractor,
        actions=[0, 1, 2],
        discount=discount,
        time_limit=time_limit,
        step_size=step_size,
        baseline_step_size=baseline_step_size,
    )


@timeout
def test_problem_6c_0(tiny_features: FeatureExtractor, initial_theta: np.ndarray):
    rl = make_policy_gradient(tiny_features)
    rl.theta = initial_theta.copy()
    gold = {
        (0.5, 0.2): np.array([0.3487821613, 0.3418758118, 0.3093420269]),
        (-1.0, 0.5): np.array([0.0674875466, 0.5242365881, 0.4082758653]),
        (0.0, 0.0): np.array([0.3375845378, 0.2500887766, 0.4123266856]),
    }

    for state, expected in gold.items():
        state = np.array(state)
        actual = rl.get_action_probs(state)

        msg = f'[Error] get_action_probs() failed for state {state}.'
        npt.assert_allclose(actual, expected, atol=1e-6, err_msg=msg)

    # With \phi((0.3, -0.2)) = [1, 0.3, -0.2, -0.06],
    # the logits are the first row of theta, [1000, 0, -1000].
    rl.theta = np.zeros((4, 3))
    rl.theta[0] = [1000.0, 0.0, -1000.0]

    actual = rl.get_action_probs(np.array([0.3, -0.2]))
    msg = '[Error] get_action_probs() failed for state (0.3, -0.2).'
    npt.assert_array_less(actual, np.inf, err_msg=msg)

    expected = np.array([1.0, 0.0, 0.0])
    npt.assert_allclose(actual, expected, atol=1e-6, err_msg=msg)


@timeout
@pytest.mark.parametrize('explore', [True, False])
def test_problem_6c_1(tiny_features: FeatureExtractor, explore: bool):
    rl = make_policy_gradient(tiny_features)

    # With phi((0, 0)) = [1, 0, 0, 0], the logits are the first row of theta,
    # so the probabilities are [0.1, 0.2, 0.7].
    state = np.array([0.0, 0.0])
    probs = np.array([0.1, 0.2, 0.7])
    rl.theta = np.zeros((4, 3))
    rl.theta[0] = np.log(probs)

    # About 5.5 standard deviations around the expected counts out of 10000
    bounds = {0: (835, 1165), 1: (1780, 2220), 2: (6747, 7253)}
    counts = {action: 0 for action in rl.actions}

    for _ in range(10000):
        action = rl.get_action(state, explore=explore)
        if action not in counts:
            raise AssertionError(
                f'`rl.get_action()` function returned an invalid action: {action}. '
                f'Valid actions are: {list(counts.keys())}.'
            )
        counts[action] += 1

    for action, (low, high) in bounds.items():
        if not low < counts[action] < high:
            raise AssertionError(
                f'With explore={explore}, action {action} has probability {probs[action]}, '
                f'but was returned {counts[action]} times out of 10000 (expected between '
                f'{low} and {high}).'
            )


@timeout
def test_problem_6c_2(tiny_features: FeatureExtractor, initial_theta: np.ndarray):
    rl = make_policy_gradient(tiny_features)

    # theta is all zeros, so every action has probability 1/3.
    expected = np.array(
        [
            [-0.3333333333, -0.3333333333, 0.6666666667],
            [-0.1666666667, -0.1666666667, 0.3333333333],
            [0.0666666667, 0.0666666667, -0.1333333333],
            [0.0333333333, 0.0333333333, -0.0666666667],
        ]
    )

    actual = rl.log_prob_gradient(np.array([0.5, -0.2]), 2)
    msg = '[Error] log_prob_gradient() failed for state (0.5, -0.2), action 2.'
    npt.assert_allclose(actual, expected, atol=1e-6, err_msg=msg)

    rl.theta = initial_theta.copy()
    gold = {
        ((0.5, 0.2), 0): np.array(
            [
                [0.6512178387, -0.3418758118, -0.3093420269],
                [0.3256089193, -0.1709379059, -0.1546710134],
                [0.1302435677, -0.0683751624, -0.0618684054],
                [0.0651217839, -0.0341875812, -0.0309342027],
            ]
        ),
        ((-1.0, 0.5), 1): np.array(
            [
                [-0.0674875466, 0.4757634119, -0.4082758653],
                [0.0674875466, -0.4757634119, 0.4082758653],
                [-0.0337437733, 0.2378817060, -0.2041379327],
                [0.0337437733, -0.2378817060, 0.2041379327],
            ]
        ),
    }

    for (state, action), expected in gold.items():
        actual = rl.log_prob_gradient(np.array(state), action)

        msg = f'[Error] log_prob_gradient() failed for state {state}, action {action}.'
        npt.assert_allclose(actual, expected, atol=1e-6, err_msg=msg)


def make_step(
    state: tuple[float, float],
    action: int,
    reward: float,
    next_state: tuple[float, float],
    terminal: bool = False,
) -> a4.MDPTransition:
    """Create an MDPTransition for the given parameters."""
    return a4.MDPTransition(
        state=np.array(state),
        action=action,
        reward=reward,
        next_state=np.array(next_state),
        terminal=terminal,
    )


@timeout
def test_problem_6c_3(tiny_features: FeatureExtractor):
    rl = make_policy_gradient(tiny_features)
    rl.rollout = [
        make_step((0.5, 0.2), 2, -1.0, (0.6, 0.1)),
        make_step((0.6, 0.1), 0, -2.0, (0.4, -0.1)),
        make_step((0.4, -0.1), 1, -0.5, (0.3, -0.2)),
    ]
    rl.update_policy()

    # On the first rollout the baseline is set to the utility,
    # so utility - baseline = 0 and theta does not change.
    if rl.baseline is None or not math.isclose(rl.baseline, -3.205, abs_tol=1e-6):
        raise AssertionError(
            f"After the first rollout, the baseline should be the rollout's utility "
            f'-3.205, but it is {rl.baseline}.'
        )

    expected = np.zeros((4, 3))
    msg = '[Error] On the first rollout, utility - baseline should be 0.'
    npt.assert_allclose(rl.theta, expected, atol=1e-6, err_msg=msg)


@timeout
def test_problem_6c_4(tiny_features: FeatureExtractor, initial_theta: np.ndarray):
    rl = make_policy_gradient(tiny_features)
    rl.theta = initial_theta.copy()
    rl.baseline = -2.0
    rl.rollout = [
        make_step((0.5, 0.2), 2, -1.0, (-1.0, 0.5)),
        make_step((-1.0, 0.5), 0, -0.5, (0.0, 0.0)),
        make_step((0.0, 0.0), 1, 2.0, (0.1, 0.1)),
    ]
    rl.update_policy()

    expected = np.array(
        [
            [0.1534136287, -0.2252156553, 0.2718020266],
            [0.2598019331, 0.0766658140, -0.3364677471],
            [-0.9139595446, 1.9282829200, -0.0143233754],
            [0.2912538259, -0.2505390353, 0.2592852094],
        ]
    )

    msg = '[Error] `update_policy()` failed to update theta correctly.\n'
    msg += '[Hint] This test uses discount=0.9, step_size=0.1, and baseline_step_size=0.25.\n'
    msg += '[Hint] `update_policy()` uses your `log_prob_gradient()`, so make sure you '
    msg += 'pass `test_problem_6c_2` first.'
    npt.assert_allclose(rl.theta, expected, atol=1e-6, err_msg=msg)

    if rl.baseline is None or not math.isclose(rl.baseline, -1.4575, abs_tol=1e-6):
        raise AssertionError(
            f'The baseline after the update should be -1.4575, but it is {rl.baseline}.'
        )


@timeout
def test_problem_6c_5(tiny_features: FeatureExtractor, initial_theta: np.ndarray):
    # The agent without a baseline in 6d: self.baseline is 0.0 (not None)
    # and self.baseline_step_size is 0.0.
    rl = make_policy_gradient(tiny_features, baseline_step_size=0.0)
    rl.theta = initial_theta.copy()
    rl.baseline = 0.0
    rl.rollout = [
        make_step((0.5, 0.2), 2, -1.0, (0.6, 0.1)),
        make_step((0.6, 0.1), 0, -2.0, (0.4, -0.1)),
        make_step((0.4, -0.1), 1, -0.5, (0.3, -0.2)),
    ]
    rl.update_policy()

    if rl.baseline is None or not math.isclose(rl.baseline, 0.0, abs_tol=1e-6):
        raise AssertionError(
            f'The baseline after the update should be 0.0, but it is {rl.baseline}.'
        )

    expected = np.array(
        [
            [0.1676095457, -0.2506639216, 0.2830543759],
            [0.5003677190, 0.0093349469, -0.5097026659],
            [-1.0110801173, 2.0565811473, -0.0455010300],
            [0.3940181900, -0.2733117776, 0.1792935876],
        ]
    )

    msg = '[Error] update_policy() failed to update theta correctly with baseline=0.\n'
    msg += '[Hint] This test uses discount=0.9, step_size=0.1, and baseline_step_size=0.0.\n'
    msg += '[Hint] Since baseline_step_size=0.0, the baseline should not change after the update.'
    npt.assert_allclose(rl.theta, expected, atol=1e-6)


@timeout
def test_problem_6c_6_hidden(tiny_features: FeatureExtractor):
    # Several rollouts through step(), some ending at a terminal state
    # and some at the time limit.
    rng = np.random.default_rng(221)
    rollouts: list[list[a4.MDPTransition]] = []

    for length, terminal in ((2, True), (4, False), (3, True), (4, False)):
        rollout: list[a4.MDPTransition] = []

        for t in range(length):
            state = rng.uniform(-1, 1, size=2)
            next_state = rng.uniform(-1, 1, size=2)

            transition = a4.MDPTransition(
                state=state,
                action=int(rng.integers(3)),
                reward=rng.uniform(-2, 1),
                next_state=next_state,
                terminal=terminal and t == length - 1,
            )
            rollout.append(transition)

        rollouts.append(rollout)

    rl = make_policy_gradient(tiny_features, time_limit=4)
    thetas: list[np.ndarray] = []
    baselines: list[float | None] = []

    for rollout in rollouts:
        for transition in rollout:
            rl.step(transition)

        thetas.append(rl.theta)
        baselines.append(rl.baseline)


def make_bootstrapped(
    feature_extractor: FeatureExtractor,
    initial_theta: np.ndarray,
    time_limit: int,
    weight: tuple[float, ...] = (-2.0, 1.0, 0.5, -1.5),
) -> a4.BootstrappedPolicyGradient:
    """Returns a ``BootstrappedPolicyGradient`` with theta and a fixed V(s). phi(s) on the
    same tiny features, so that a bug in V does not affect the tests.
    """
    rl = a4.BootstrappedPolicyGradient(
        feature_dim=4,
        feature_extractor=feature_extractor,
        actions=[0, 1, 2],
        discount=0.9,
        time_limit=time_limit,
        step_size=0.1,
    )
    rl.theta = initial_theta.copy()
    rl.value_function.weight = np.array(weight)
    return rl


def check_bootstrapped(
    feature_extractor: FeatureExtractor,
    bootstrap_states: list[tuple[float, float]],
    bootstrap_actions: list[int],
    bootstrap_rewards: list[float],
    initial_theta: np.ndarray,
    expected: np.ndarray,
    terminal: bool,
) -> None:
    """Check that ``BootstrappedPolicyGradient.update_policy()`` updates theta correctly for
    a 3-step rollout, either ending at a terminal state or at the time limit."""
    # A 3-step rollout: at the flag before the time limit (1000),
    # or at the time limit (3) without reaching the flag.
    time_limit = 1000 if terminal else 3
    rl = make_bootstrapped(feature_extractor, initial_theta, time_limit=time_limit)
    rl.rollout = [
        make_step(
            bootstrap_states[t],
            bootstrap_actions[t],
            bootstrap_rewards[t],
            bootstrap_states[t + 1],
            terminal and t == 2,
        )
        for t in range(3)
    ]
    rl.update_policy()

    msg = '[Error] `update_policy()` failed to update theta correctly.\n'
    msg += '[Hint] This test uses discount=0.9, step_size=0.1, and weight=[-2, 1, 0.5, -1.5].\n'
    msg += '[Hint] `update_policy()` uses your `log_prob_gradient()`, so make sure you '
    msg += '[Hint] pass `test_problem_6c_2` first.'
    npt.assert_allclose(rl.theta, expected, atol=1e-6, err_msg=msg)


@timeout
def test_problem_6f_0(
    tiny_features: FeatureExtractor,
    initial_theta: np.ndarray,
    bootstrap_states: list[tuple[float, float]],
    bootstrap_actions: list[int],
    bootstrap_rewards: list[float],
):
    # The rollout ends at the flag: the last step's next state is terminal.
    expected = np.array(
        [
            [-0.0194114186, 0.1584260635, 0.0609853551],
            [0.5497742587, 0.0056401406, -0.5554143993],
            [-1.0052681328, 2.0164104441, -0.0111423113],
            [0.4183474638, -0.3035901012, 0.1852426374],
        ]
    )

    check_bootstrapped(
        tiny_features,
        bootstrap_states,
        bootstrap_actions,
        bootstrap_rewards,
        initial_theta,
        expected,
        terminal=True,
    )


@timeout
def test_problem_6f_1(
    tiny_features: FeatureExtractor,
    initial_theta: np.ndarray,
    bootstrap_states: list[tuple[float, float]],
    bootstrap_actions: list[int],
    bootstrap_rewards: list[float],
):
    # The rollout ends at the time limit, so the last step's next state is not terminal.
    expected = np.array(
        [
            [0.0328466679, 0.0423398061, 0.1248135260],
            [0.5497742587, 0.0056401406, -0.5554143993],
            [-1.0052681328, 2.0164104441, -0.0111423113],
            [0.4183474638, -0.3035901012, 0.1852426374],
        ]
    )

    check_bootstrapped(
        tiny_features,
        bootstrap_states,
        bootstrap_actions,
        bootstrap_rewards,
        initial_theta,
        expected,
        terminal=False,
    )
