import json
from pathlib import Path

import numpy as np

import dnnlpy.cs221.assignment4 as a4

from ..utils import timeout

FIXTURES_PATH = Path(__file__).parent / 'fixtures'


def run_vi_over_number_line(mdp: a4.NumberLineMDP) -> np.ndarray:
    """Builds the T and R tensors for the NumberLineMDP and runs VI on them."""
    actions = np.array(mdp.actions, dtype=np.int8)
    transitions = np.zeros((mdp.num_states, len(actions), mdp.num_states))
    rewards = np.zeros_like(transitions)
    valid_actions = np.zeros((mdp.num_states, len(actions)), dtype=np.bool)

    for state_idx, state in enumerate(mdp.indexer.all_states()):
        if state in mdp.terminal_states:
            continue

        valid_actions[state_idx, :] = True
        for action_idx, action in enumerate(actions):
            forward_prob = 0.2 if action == 1 else 0.3
            backward_prob = 1.0 - forward_prob
            forward_state = state + 1
            backward_state = state - 1

            forward_reward = (
                mdp.right_reward
                if forward_state == mdp.n
                else (mdp.left_reward if forward_state == -mdp.n else mdp.penalty)
            )
            backward_reward = (
                mdp.right_reward
                if backward_state == mdp.n
                else (mdp.left_reward if backward_state == -mdp.n else mdp.penalty)
            )

            forward_idx = mdp.state_to_index(forward_state)
            backward_idx = mdp.state_to_index(backward_state)
            transitions[state_idx, action_idx, forward_idx] += forward_prob
            rewards[state_idx, action_idx, forward_idx] = forward_reward
            transitions[state_idx, action_idx, backward_idx] += backward_prob
            rewards[state_idx, action_idx, backward_idx] = backward_reward

    vi = a4.ValueIteration(transitions, rewards, mdp.discount, 0.001, valid_actions)
    pi = vi.compute_policy()
    return pi


@timeout
def test_problem_3a_0():
    mdp = a4.NumberLineMDP()
    pi = run_vi_over_number_line(mdp)
    gold = {-1: 1, 0: 2, 1: 2}

    for key, expected in gold.items():
        idx = mdp.state_to_index(key)
        if not mdp.actions[pi[idx]] == expected:
            raise AssertionError(
                f'Incorrect pi for the state {key}: expected {expected}, '
                f'got {mdp.actions[pi[idx]]}.'
            )


@timeout
def test_problem_3a_1():
    mdp = a4.NumberLineMDP(20, left_reward=10, right_reward=30, penalty=-1)
    pi = run_vi_over_number_line(mdp)
    with open(FIXTURES_PATH / '3a-1-gold.json') as fp:
        gold: dict[str, int] = json.load(fp)

    for key, expected in gold.items():
        idx = mdp.state_to_index(int(key))
        if not mdp.actions[pi[idx]] == expected:
            raise AssertionError(
                f'Incorrect pi for the state {key}: expected {expected}, '
                f'got {mdp.actions[pi[idx]]}.'
            )

    # Test a deterministic MDP with gamma=0.9 and terminal state 2.
    # - State 0: a0 -> state 1 (r=0), a1 -> state 2 (r=5); optimal action is a1.
    # - State 1: only a1 is valid (r=-2); a0 is invalid despite r=10.
    # - State 3: only a0 is valid (r=-1); a1 has no transitions and must be masked.
    # - State 4: a0 -> state 3 (Q=-0.9), a1 -> state 2 (Q=-0.95);
    #   optimal action is a0, verifying correct discounting.
    # Verifies action masking and discount factor handling.
    P = np.zeros((5, 2, 5))
    R = np.zeros((5, 2, 5))

    for T in (
        a4.MDPTransition(0, 0, 0, 1, True),
        a4.MDPTransition(0, 1, 5, 2, True),
        a4.MDPTransition(1, 0, 10, 2, True),
        a4.MDPTransition(1, 1, -2, 2, True),
        a4.MDPTransition(3, 0, -1, 2, True),
        a4.MDPTransition(4, 0, 0, 3, True),
        a4.MDPTransition(4, 1, -0.95, 2, True),
    ):
        P[T.state, T.action, T.next_state] = 1.0
        R[T.state, T.action, T.next_state] = T.reward

    valid_actions = np.array(
        [
            [True, True],
            [False, True],
            [False, False],
            [True, False],
            [True, True],
        ],
        dtype=np.bool_,
    )

    vi = a4.ValueIteration(P, R, 0.9, valid_actions=valid_actions)
    pi = vi.compute_policy()

    for state, expected in enumerate([1, 1, -1, 0, 0]):
        if pi[state] != expected:
            raise AssertionError(
                f'Incorrect pi for the state {state}: expected {expected}, '
                f'got {pi[state]}.'
            )


@timeout(seconds=15)
def test_problem_3a_2_hidden():
    mdp = a4.NumberLineMDP(n=200)
    run_vi_over_number_line(mdp)


def make_rl(
    mdp: a4.NumberLineMDP,
    calc_val_iter_every: int = 10000,
    epsilon: float = 0.2,
    seed: int | None = None,
) -> a4.ModelBasedMonteCarlo[int, int]:
    """Make a ModelBasedMonteCarlo instance for the given MDP."""
    return a4.ModelBasedMonteCarlo(
        mdp.actions,
        mdp.discount,
        mdp.num_states,
        mdp.state_to_index,
        calc_val_iter_every=calc_val_iter_every,
        epsilon=epsilon,
        seed=seed,
    )


@timeout
def test_problem_3b_0():
    mdp = a4.NumberLineMDP()
    rl = make_rl(mdp, calc_val_iter_every=1)

    rl.pi_indices = np.full(mdp.num_states, -1, dtype=np.int8)
    for state, action in {-1: 1, 0: 2, 1: 2}.items():
        state = mdp.state_to_index(state)
        action = rl.actions.index(action)
        rl.pi_indices[state] = action

    rl.num_iters = int(2e4)
    counter = {-1: 0, 0: 0, 1: 0}

    for _ in range(10000):
        for state in range(-mdp.n + 1, mdp.n):
            idx = mdp.state_to_index(state)
            action = rl.get_action(state)
            if rl.actions[rl.pi_indices[idx]] == action:
                counter[state] += 1

    for key, value in counter.items():
        if value < 8800:
            raise AssertionError(
                f'Too few optimal actions returned for the state {key}: {value}.'
            )
        if value > 9200:
            raise AssertionError(
                f'Too much random actions returned for the state {key}: {value}.'
            )


@timeout
def test_problem_3b_1():
    mdp = a4.NumberLineMDP()

    rl = make_rl(mdp, calc_val_iter_every=100)
    rl.num_iters = 1
    rl.step(a4.MDPTransition(1, 1, 50, 2, True))
    rl.step(a4.MDPTransition(1, 1, -5, 0, False))
    rl.num_iters = 100
    rl.step(a4.MDPTransition(-1, 2, 10, -2, True))

    gold = {-1: 2, 1: 1}
    for state, expected in gold.items():
        idx = mdp.state_to_index(state)
        action = rl.actions[rl.pi_indices[idx]]

        if not action == expected:
            raise AssertionError(
                f'Incorrect policy for state {state}: expected {expected}, got {action}.'
            )

    # In state 1, action 1 led to state 0 once with reward -5,
    # and action 2 led to state 0 once and to state 2 once,
    # with reward -8 each time. State 0 has no experience yet,
    # so it counts as an end state with value 0.
    # The estimated Q-values are -5 for action 1,
    # 0.5 * -8 + 0.5 * -8 = -8 for action 2,
    # so the policy picks action 1.
    rl = make_rl(mdp, calc_val_iter_every=100)
    rl.num_iters = 1
    rl.step(a4.MDPTransition(1, 1, -5, 0, False))
    rl.step(a4.MDPTransition(1, 2, -8, 0, False))
    rl.step(a4.MDPTransition(1, 2, -8, 2, True))
    rl.num_iters = 100
    rl.step(a4.MDPTransition(-1, 1, -5, 0, False))

    idx = mdp.state_to_index(1)
    if not rl.actions[rl.pi_indices[idx]] == 1:
        raise AssertionError(
            'Incorrect policy for state 1 when an action has several next states. '
            "Check how you estimate the average reward of each `(s, a, s')`."
        )


@timeout
def test_problem_3b_2():
    mdp = a4.NumberLineMDP()
    rl = make_rl(mdp, calc_val_iter_every=100)
    rl.num_iters = 1
    rl.step(a4.MDPTransition(0, 1, -5, 1, False))
    rl.step(a4.MDPTransition(0, 1, -5, 1, False))
    rl.step(a4.MDPTransition(0, 1, -5, -1, False))
    rl.step(a4.MDPTransition(0, 2, -5, 1, False))
    rl.step(a4.MDPTransition(0, 2, -5, -1, False))
    rl.step(a4.MDPTransition(1, 1, 50, 2, True))
    rl.step(a4.MDPTransition(1, 1, 50, 2, True))
    rl.step(a4.MDPTransition(1, 1, -5, 0, False))
    rl.step(a4.MDPTransition(1, 2, 50, 2, True))
    rl.step(a4.MDPTransition(1, 2, -5, 0, False))
    rl.step(a4.MDPTransition(-1, 1, -5, 0, False))
    rl.step(a4.MDPTransition(-1, 2, -5, 0, False))
    rl.num_iters = 100
    rl.step(a4.MDPTransition(-1, 2, 10, -2, True))

    gold = {-1: 1, 0: 1, 1: 1}
    for state, expected in gold.items():
        idx = mdp.state_to_index(state)
        actual = rl.actions[rl.pi_indices[idx]]
        if not actual == expected:
            raise AssertionError(
                f'Incorrect policy for the state {state} after MC value iteration: '
                f'expected {expected}, got {actual}.'
            )


@timeout
def test_problem_3b_3():
    mdp = a4.NumberLineMDP()
    rl = make_rl(mdp, calc_val_iter_every=100)

    counts = {}
    for state in mdp.all_states:
        counts[state] = 0

    for _ in range(10000):
        for state in counts:
            action = rl.get_action(state)
            if action == 1:
                counts[state] += 1

    if counts[-mdp.n] < 4000 or counts[mdp.n] > 6000:
        raise AssertionError(
            f'Incorrect action distribution for state {-mdp.n}: {counts[-mdp.n]}'
        )
