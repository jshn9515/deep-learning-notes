import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

import numpy as np

from .mdp import MDPTransition

type Axis = int | tuple[int, ...] | None

__all__ = [
    'BootstrappedPolicyGradient',
    'ConstrainedQLearning',
    'FixedRLAlgorithm',
    'FunctionApproxQLearning',
    'ModelBasedMonteCarlo',
    'PolicyGradient',
    'RLAlgorithm',
    'RandomRLAlgorithm',
    'TabularQLearning',
    'ValueIteration',
]


def softmax(x: np.ndarray, axis: Axis = None) -> np.ndarray:
    """Compute the softmax of the input array along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


class RLAlgorithm[StateT, ActionT](ABC):
    """An abstract base class for reinforcement learning algorithms.

    Defines the interface for selecting actions and learning from environment feedback.
    The simulator (see :func:`simulate()`) will call :meth:`get_action()` to get an action,
    perform the action, and then provide feedback (via :meth:`step()`) to the RL algorithm,
    so it can adjust its parameters.
    """

    @abstractmethod
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        """Select an action for the given state.

        Args:
            state (StateT): The current state.
            explore (bool, optinal): Whether to allow exploration during action selection.
                Defaults to ``True``.

        Returns:
            ActionT: The selected action.
        """

    @abstractmethod
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        """Update the algorithm using an observed environment transition.

        Args:
            transition (MDPTransition): A tuple containing the current step's transition
                information: state, action, reward, next_state, and terminal flag.
        """


class FixedRLAlgorithm[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    """A RL agent that follows a fixed policy without learning.

    Args:
        pi (dict[StateT, ActionT]): A dictionary mapping states to actions.
        actions (list[ActionT]): A list of possible actions.
        epsilon (float, optional): The probability of taking a random action instead of
            the one specified by the policy :math:`pi`. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        pi: dict[StateT, ActionT],
        actions: list[ActionT],
        epsilon: float = 0.2,
        seed: int | None = None,
    ) -> None:
        r"""Initialize the ``FixedRLAlgorithm`` with a fixed policy :math:`\pi`."""
        self.pi = pi
        self.actions = list(dict.fromkeys(actions))
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    @override
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        r"""Returns the action for the given state according to the fixed policy :math:`\pi`.
        If :attr:`explore` is ``True``, it may return a random action with probability
        :attr:`epsilon`.

        Args:
            state (StateT): The current state of the MDP.
            explore (bool, optional): Whether the algorithm should explore or not.
                Defaults to ``True``.

        Returns:
            ActionT: The action to take in the given state.
        """
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        else:
            return self.pi[state]

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        """This method does nothing because the :class:`FixedRLAlgorithm` does not learn
        from feedback.
        """


class RandomRLAlgorithm[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    """A RL agent that selects actions uniformly at random without learning.

    Args:
        actions (list[ActionT]): A list of possible actions.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(self, actions: list[ActionT], seed: int | None = None) -> None:
        """Initialize the ``RandomRLAlgorithm`` with a list of possible actions."""
        self.actions = list(dict.fromkeys(actions))
        self.rng = random.Random(seed)

    @override
    def get_action(self, state: StateT, explore: bool = False) -> ActionT:
        """Returns a random action from the list of available actions.

        Args:
            state (StateT): The current state of the MDP. This argument is ignored.
            explore (bool, optional): Whether the algorithm should explore or not.
                Defaults to ``False``. This argument is ignored.

        Returns:
            ActionT: A randomly selected action from the list of available actions.
        """
        return self.rng.choice(self.actions)

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        """This method does nothing because the :class:`RandomRLAlgorithm` does not learn
        from feedback.
        """


# Problem 3a: Value Iteration on Number Line (from Problem 1)
class ValueIteration:
    r"""A RL algorithm that computes the optimal policy using value iteration.

    Args:
        transitions (np.ndarray): Transition probabilities with shape ``(num_states,
            num_actions, num_states)``.
        rewards (np.ndarray): Rewards with shape ``(num_states, num_actions, num_states)``.
        discount (float): Discount factor :math:`\gamma` for future rewards.
        epsilon (float, optional): Convergence threshold for value iteration. Defaults to 0.001.
        valid_actions (np.ndarray, optional): Boolean mask indicating valid actions for each
            state with shape ``(num_states, num_actions)``. Defaults to ``None``.
    """

    def __init__(
        self,
        transitions: np.ndarray,
        rewards: np.ndarray,
        discount: float,
        epsilon: float = 0.001,
        valid_actions: np.ndarray | None = None,
    ):
        """Initialize the value iteration algorithm with the specified parameters."""
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        self.epsilon = epsilon
        self.valid_actions = valid_actions

        self.num_states = transitions.shape[0]
        self.num_actions = transitions.shape[1]

        if valid_actions is None:
            action_mask = np.ones((self.num_states, self.num_actions), dtype=np.bool_)
        else:
            action_mask = np.asarray(valid_actions, dtype=np.bool_)

        self.action_mask = action_mask
        self.values = np.zeros(self.num_states)
        self.has_valid_actions = np.any(action_mask, axis=1)

    def compute_q_values(self) -> np.ndarray:
        r"""Compute Q-values for all state-action pairs using the current value function.

        The Q-value for a state-action pair is defined as the expected return of taking
        that action in that state and following the current value function thereafter:

        .. math::
            Q(s,a) = \sum_{s'} P(s'\mid s,a)[R(s,a,s') + \gamma V(s')]

        The invalid actions are masked by setting their Q-values to negative infinity.

        Returns:
            np.ndarray: Q-values with shape ``(num_states, num_actions)``.
        """
        q_values = np.sum(
            self.transitions
            * (self.rewards + self.discount * self.values[None, None, :]),
            axis=2,
        )
        q_values = np.where(self.action_mask, q_values, -np.inf)
        return q_values

    def compute_policy(self) -> np.ndarray:
        r"""Compute the optimal policy using value iteration.

        Iteratively updates the value function using the Bellman optimality equation until
        the maximum change in state values is below :attr:`self.epsilon`. The policy is
        then extracted by selecting the action with the highest Q-value in each state.

        The optimal value function and policy are defined as:

        .. math::

            \begin{aligned}
                V^*(s) &= \max_a Q^*(s,a) \\
                \pi^*(s) &= \arg\max_a Q^*(s,a)
            \end{aligned}

        States without valid actions keep value zero and receive policy index ``-1``.

        Returns:
            np.ndarray: Optimal action indices for each state with shape ``(num_states,)``.
        """
        while True:
            q_values = self.compute_q_values()
            new_values = np.max(q_values, axis=1)
            new_values = np.where(self.has_valid_actions, new_values, 0.0)

            delta = np.max(np.abs(new_values - self.values))
            self.values = new_values

            if delta < self.epsilon:
                break

        q_values = self.compute_q_values()
        new_policy = np.argmax(q_values, axis=1)
        new_policy = np.where(self.has_valid_actions, new_policy, -1)
        return new_policy


# Problem 3b: Model-Based Monte Carlo
class ModelBasedMonteCarlo[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    """A model-based RL agent that learns an environment model from observed transitions.

    The agent maintains empirical estimates of transition dynamics and rewards. At
    configurable intervals, value iteration is applied to the estimated model to improve
    its policy.

    The agent uses an epsilon-greedy strategy for action selection, balancing exploration
    and exploitation during training.

    Args:
        actions (list[ActionT]): A list of possible actions.
        discount (float): Discount factor for future rewards, between 0 and 1.
        num_states (int): Total number of discrete states available.
        state_to_index (Callable[[StateT], int]): Function mapping states to contiguous
            indices ``[0, num_states)``.
        index_to_state (Callable[[int], StateT], optional): The inverse of
            ``state_to_index``. Defaults to the identity function.
        calc_val_iter_every (int, optional): Run value iteration every this many steps.
            Defaults to 10000.
        epsilon (float, optional): The epsilon value indicating how frequently the policy
            returns a random action. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        actions: list[ActionT],
        discount: float,
        num_states: int,
        state_to_index: Callable[[StateT], int],
        index_to_state: Callable[[int], StateT] = lambda idx: idx,
        calc_val_iter_every: int = 10000,
        epsilon: float = 0.2,
        seed: int | None = None,
    ):
        """Initialize the ``ModelBasedMonteCarlo`` agent with the specified parameters."""
        self.actions = list(dict.fromkeys(actions))
        self.discount = discount
        self.num_states = num_states
        self.state_to_index = state_to_index
        self.index_to_state = index_to_state
        self.calc_val_iter_every = calc_val_iter_every
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.num_iters = 0
        self.num_actions = len(self.actions)

        # N: [s, a, s'] -> number of times we observed that transition
        self.transition_counts = np.zeros((num_states, self.num_actions, num_states))
        # R: [s, a, s'] -> sum of rewards observed for that transition
        self.reward_sums = np.zeros((num_states, self.num_actions, num_states))
        # S: [s, a] -> whether we have ever observed that (s, a) pair
        self.valid_actions = np.zeros((num_states, self.num_actions), dtype=np.bool)

        # pi_indices: [s] -> index of the action in self.actions for state s, or -1 if no action
        self.pi_indices = np.full(num_states, -1, dtype=np.int8)

    @override
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        """Select an action for the given state using an epsilon-greedy policy.

        During training, the agent balances exploration and exploitation. With probability
        :attr:`self.epsilon`, a random action is selected; otherwise, the action specified
        by the current policy is chosen.

        During evaluation, the agent follows the current policy without exploration. If
        no action has been assigned to the given state, a random action is selected as a
        fallback.

        The exploration probability may be adjusted dynamically according to a predefined
        exploration schedule.

        Args:
            state (StateT): The current environment state.
            explore (bool, optional): Whether to enable exploration during action selection.
                Defaults to ``True``.

        Returns:
            ActionT: The action selected by the agent.
        """
        if explore:
            self.num_iters += 1

        if self.num_iters < 2e4:  # Always explore
            epsilon = 1.0
        elif 2e4 <= self.num_iters < 1e6:  # Use the default eps
            epsilon = self.epsilon
        else:  # >1e6
            epsilon = self.epsilon / math.log(self.num_iters - 1e6 + 1)  # decay

        state_idx = self.state_to_index(state)
        policy_idx = self.pi_indices[state_idx]

        if policy_idx == -1 or (explore and self.rng.random() < epsilon):
            return self.rng.choice(self.actions)

        return self.actions[policy_idx]

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        """Update the empirical environment model using an observed transition.

        Records the transition count and accumulated reward for each ``(state, action,
        next_state)`` tuple.

        Periodically estimates transition probabilities and expected rewards from the
        collected observations, then runs value iteration on the estimated model to
        update the policy.

        The resulting policy is stored as action indices in self.pi_indices, with ``-1``
        indicating states without a valid action.

        Args:
            transition (MDPTransition): A tuple containing the current step's transition
                information: state, action, reward, next_state, and terminal flag.
        """
        s = self.state_to_index(transition.state)
        a = self.actions.index(transition.action)
        ns = self.state_to_index(transition.next_state)

        self.transition_counts[s, a, ns] += 1
        self.reward_sums[s, a, ns] += transition.reward
        self.valid_actions[s, a] = True

        if self.num_iters > 0 and self.num_iters % self.calc_val_iter_every == 0:
            # 1. Count total transitions for each (state, action)
            total_counts = np.sum(self.transition_counts, axis=2, keepdims=True)

            # 2. Estimate transition probabilities
            transition_probs = np.divide(
                self.transition_counts,
                total_counts,
                out=np.zeros_like(self.transition_counts),
                where=total_counts > 0,
            )

            # 3. Estimate average rewards
            average_rewards = np.divide(
                self.reward_sums,
                self.transition_counts,
                out=np.zeros_like(self.reward_sums),
                where=self.transition_counts > 0,
            )

            # 4. Run value iteration to update the policy
            vi = ValueIteration(
                transitions=transition_probs,
                rewards=average_rewards,
                discount=self.discount,
                valid_actions=self.valid_actions,
            )
            self.pi_indices = vi.compute_policy()


# Problem 4a: tabular Q-learning
class TabularQLearning[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    """A model-free RL agent using tabular Q-learning.

    The agent maintains a Q-table that estimates the optimal action-value function for
    each state-action pair. Q-values are updated incrementally using observed transitions
    and the Bellman optimality equation.

    The agent uses an epsilon-greedy strategy for action selection, balancing exploration
    and exploitation during training. The exploration probability is adjusted dynamically
    according to a predefined schedule.

    Args:
        actions (list[ActionT]): A list of possible actions.
        discount (float): Discount factor for future rewards, between 0 and 1.
        num_states (int): Total number of discrete states available.
        state_to_index (Callable[[StateT], int]): Function mapping states to contiguous
            indices ``[0, num_states)``.
        epsilon (float, optional): The epsilon value indicating how frequently the agent
            selects a random action. Defaults to 0.2.
        initial_q (float, optional): Initial value of all Q-table entries. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        actions: list[ActionT],
        discount: float,
        num_states: int,
        state_to_index: Callable[[StateT], int],
        epsilon: float = 0.2,
        initial_q: float = 0.0,
        seed: int | None = None,
    ):
        """Initialize the ``TabularQLearning`` agent with the specified parameters."""
        self.actions = list(dict.fromkeys(actions))
        self.discount = discount
        self.num_states = int(num_states)
        self.state_to_index = state_to_index
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # self.Q[s, a] is the current estimate of Q* for state s and action a,
        # where s = self.state_to_index(state) and a = self.actions.index(action).
        self.Q = np.full((self.num_states, len(self.actions)), float(initial_q))
        self.num_iters = 0

    @override
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        """Select an action for the given state using an epsilon-greedy policy.

        During training, the agent balances exploration and exploitation. With probability
        :attr:`self.epsilon`, a random action is selected; otherwise, the action with the
        highest estimated Q-value is chosen.

        During evaluation, the agent selects the action with the highest estimated Q-value
        without exploration. Ties are resolved by selecting the first action with the
        maximum Q-value.

        The exploration probability is adjusted dynamically according to a predefined
        schedule, with full exploration during the initial steps and logarithmic decay after
        a specified number of iterations.

        Args:
            state (StateT): The current environment state.
            explore (bool, optional): Whether to enable exploration during action selection.
                Defaults to ``True``.

        Returns:
            ActionT: The action selected by the agent.
        """
        if explore:
            self.num_iters += 1

        if self.num_iters < 2e4:  # Always explore
            epsilon = 1.0
        elif 2e4 <= self.num_iters < 1e5:  # Use the default eps
            epsilon = self.epsilon
        else:  # >1e5
            epsilon = self.epsilon / math.log(self.num_iters - 1e5 + 1)  # decay

        state_idx = self.state_to_index(state)
        if explore and self.rng.random() < epsilon:
            return self.rng.choice(self.actions)
        else:
            action_idx = int(np.argmax(self.Q[state_idx]))
            return self.actions[action_idx]

    @property
    def step_size(self) -> float:
        """The step size for updating the Q-values."""
        return 0.1

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        r"""Update the Q-table using an observed transition.

        Applies the one-step Q-learning update rule to the observed state-action pair:

        .. math::

            Q(s, a) \leftarrow Q(s, a) + \alpha
            [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]

        where :math:`\alpha` is the learning rate, :math:`\gamma` is the discount factor,
        and :math:`r` is the observed immediate reward.

        For terminal transitions, the future reward term is omitted, so  the update target
        is simply the observed reward.

        Unlike model-based methods, Q-learning updates its Q-values directly from sampled
        transitions without estimating transition probabilities or maintaining an explicit
        environment model.

        Args:
            transition (MDPTransition): An observed transition containing the current state,
                action, reward, next state, and terminal indicator.
        """
        s = self.state_to_index(transition.state)
        a = self.actions.index(transition.action)
        ns = self.state_to_index(transition.next_state)

        if transition.terminal:
            target = transition.reward
        else:
            target = transition.reward + self.discount * np.max(self.Q[ns])

        self.Q[s, a] += self.step_size * (target - self.Q[s, a])


# Problem 4c: Q-learning with Function Approximation
class FunctionApproxQLearning[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    r"""A model-free RL agent using linear function approximation.

    The agent approximates the optimal action-value function using a linear combination
    of state features and action-specific weights, rather than maintaining an explicit
    Q-table.

    The agent uses an epsilon-greedy strategy for action selection, balancing exploration
    and exploitation during training. Both the exploration probability and learning rate
    follow predefined decay schedules.

    Args:
        feature_dim (int): Dimensionality of the state feature vectors.
        feature_extractor (Callable[[StateT], np.ndarray]): Function mapping states to
            feature vectors of shape ``(feature_dim,)``.
        actions (list[ActionT]): A list of possible actions.
        discount (float): Discount factor :math:`\gamma` for future rewards.
        epsilon (float, optional): Probability of selecting a random action. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_extractor: Callable[[StateT], np.ndarray],
        actions: list[ActionT],
        discount: float,
        epsilon: float = 0.2,
        seed: int | None = None,
    ):
        """Initialize the ``FunctionApproxQLearning`` agent with the specified parameters."""
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.actions = list(dict.fromkeys(actions))
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # `self.weight` has shape (feature_dim, num_actions).
        # Column self.weight[:, a] holds the weights for action a.
        # The mountain car actions are 0, 1, 2, so each action is also its own column index.
        self.weight = self.rng.standard_normal((feature_dim, len(actions)))
        self.num_iters = 0

    def get_q(self, state: StateT, action: ActionT) -> float:
        r"""Estimate the Q-value for a state-action pair.

        Computes the action value using linear function approximation:

        .. math::

            Q(s, a; \mathbf{w}) = \phi(s)^\top \mathbf{w}_a

        where :math:`\phi(s)` is the state feature vector and :math:`\mathbf{w}_a` is the
        weight vector for action :math:`a`.

        Args:
            state (StateT): The current environment state.
            action (ActionT): The action to evaluate.

        Returns:
            float: The estimated Q-value.
        """
        phi = self.feature_extractor(state)
        idx = self.actions.index(action)
        q_value = np.dot(phi, self.weight[:, idx])
        return float(q_value)

    @override
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        """Select an action for the given state using an epsilon-greedy policy.

        During training, the agent balances exploration and exploitation. With probability
        ``epsilon``, a random action is selected; otherwise, the action with the highest
        estimated Q-value is chosen. During evaluation, the agent always selects the action
        with the highest estimated Q-value without exploration.

        The exploration probability follows a predefined schedule, with full exploration
        during the initial steps, followed by a constant probability and logarithmic decay.

        Args:
            state (StateT): The current environment state.
            explore (bool, optional): Whether to enable exploration during action selection.
                Defaults to ``True``.

        Returns:
            ActionT: The action selected by the agent.
        """
        if explore:
            self.num_iters += 1

        if self.num_iters < 2e4:  # Always explore
            epsilon = 1.0
        elif 2e4 <= self.num_iters < 1e5:  # Use the default eps
            epsilon = self.epsilon
        else:  # >1e5
            epsilon = self.epsilon / math.log(self.num_iters - 1e5 + 1)  # decay

        if explore and self.rng.random() < epsilon:
            actions = np.array(self.actions)
            return self.rng.choice(actions)
        else:
            q_values = np.array([self.get_q(state, action) for action in self.actions])
            return self.actions[np.argmax(q_values)]

    @property
    def step_size(self) -> float:
        """The step size for updating the Q-values."""
        return 0.005 * pow(0.99, self.num_iters / 500)

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        r"""Update the action-specific weights using an observed transition.

        Applies the semi-gradient Q-learning update rule:

        .. math::

            \mathbf{w}_a \leftarrow \mathbf{w}_a + \alpha
            [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \phi(s)

        where :math:`\alpha` is the learning rate, :math:`\gamma` is the discount factor,
        and :math:`\phi(s)` is the state feature vector.

        For terminal transitions, the future reward term is omitted, and the update target
        is simply the observed reward.

        Unlike tabular Q-learning, the agent updates action-specific weights instead of
        individual Q-table entries, allowing learned values to generalize across states with
        similar features.

        Args:
            transition (MDPTransition): An observed transition containing the current state,
                action, reward, next state, and terminal indicator.
        """
        T = transition
        idx = self.actions.index(T.action)
        features = self.feature_extractor(T.state)

        old_value = self.get_q(T.state, T.action)
        if transition.terminal:
            new_value = T.reward
        else:
            new_value = T.reward + self.discount * max(
                self.get_q(T.next_state, action) for action in self.actions
            )

        self.weight[:, idx] += self.step_size * (new_value - old_value) * features


class ConstrainedQLearning(FunctionApproxQLearning[np.ndarray, int]):
    r"""A Q-learning agent with action constraints based on MountainCar dynamics.

    Extends linear function approximation Q-learning by restricting actions according
    to the predicted next velocity of the car.

    The agent uses an epsilon-greedy strategy for action selection, considering only
    actions that satisfy the velocity constraint. If no valid action is available, the
    agent returns ``None``.

    Args:
        feature_dim (int): Dimensionality of the state feature vectors.
        feature_extractor (Callable[[np.ndarray], np.ndarray]): Function mapping states
            to feature vectors of shape ``(feature_dim,)``.
        actions (list[int]): A list of possible actions.
        discount (float): Discount factor :math:`\gamma` for future rewards.
        epsilon (float, optional): Probability of selecting a random action. Defaults to 0.2.
        force (float, optional): Engine force in the MountainCar velocity update. Defaults
            to 0.001.
        gravity (float, optional): Gravity coefficient in the MountainCar velocity update.
            Defaults to 0.0025.
        max_speed (float, optional): Maximum allowed absolute next velocity. If ``None``,
            all actions are allowed. Defaults to ``None``.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_extractor: Callable[[np.ndarray], np.ndarray],
        actions: list[int],
        discount: float,
        epsilon: float = 0.2,
        force: float = 0.001,
        gravity: float = 0.0025,
        max_speed: float | None = None,
        seed: int | None = None,
    ):
        """Initialize the ``ConstrainedQLearning`` agent with the specified parameters."""
        super().__init__(
            feature_dim,
            feature_extractor,
            actions=actions,
            discount=discount,
            epsilon=epsilon,
            seed=seed,
        )
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    @override
    def get_action(self, state: np.ndarray, explore: bool = True) -> int | None:
        r"""Select an action using an epsilon-greedy policy with velocity constraints.

        The state contains the car's position and velocity, while actions ``0``, ``1``,
        and ``2`` correspond to accelerating left, coasting, and accelerating right,
        respectively.

        For each action, the next velocity is predicted using the MountainCar dynamics:

        .. math::

            v_{t+1} = v_t + (a_t - 1) \cdot F - G \cos(3p_t)

        where :math:`p_t` and :math:`v_t` are the current position and velocity, and
        :math:`F` and :math:`G` are the force and gravity coefficients.

        An action is valid if :math:`|v_{t+1}| < v_{\max}`. When no speed limit is specified,
        all actions are considered valid.

        During training, the agent selects a random valid action with probability ``epsilon``;
        otherwise, it selects the valid action with the highest estimated Q-value. During
        evaluation, exploration is disabled.

        The exploration probability follows a predefined schedule, with full exploration
        during the initial steps and logarithmic decay after a specified number of iterations.

        Args:
            state (np.ndarray): Current state containing position and velocity, with shape
                ``(2,)``.
            explore (bool, optional): Whether to enable exploration during action selection.
                Defaults to ``True``.

        Returns:
            int | None: The selected valid action, or ``None`` if no valid action is available.
        """
        if explore:
            self.num_iters += 1

        if self.num_iters < 2e4:  # Always explore
            epsilon = 1.0
        elif 2e4 <= self.num_iters < 1e5:  # Use the default eps
            epsilon = self.epsilon
        else:  # >1e5
            epsilon = self.epsilon / math.log(self.num_iters - 1e5 + 1)  # decay

        position, velocity = map(float, state)
        valid_actions: list[int] = []

        for action in self.actions:
            next_velocity = (
                velocity
                + (action - 1) * self.force
                - math.cos(3 * position) * self.gravity
            )

            if self.max_speed is None or abs(next_velocity) < self.max_speed:
                valid_actions.append(action)

        if not valid_actions:
            return None

        if explore and self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions))
        else:
            return max(valid_actions, key=lambda action: self.get_q(state, action))


class LinearValueFunction[StateT]:
    r"""A linear state-value function approximator used by :class:`BootstrappedPolicyGradient`.

    The estimated state value is defined as:

    .. math::

        \hat{V}(s) = \mathbf{w}^{\top}\phi(s)

    where :math:`\mathbf{w}` is the weight vector and :math:`\phi(s)` is the feature
    representation of state :math:`s`.

    The weights are updated using semi-gradient TD(0), following the same update rule as
    :class:`FunctionApproxQLearning` in Problem 4c, with :math:`\hat{V}` replacing
    :math:`\hat{Q}`:

    .. math::

        \mathbf{w} \leftarrow \mathbf{w} +
        \alpha \left(r + \gamma\hat{V}(s') - \hat{V}(s)\right)\phi(s)

    where:

    - :math:`\alpha` is the learning rate (``step_size``).
    - :math:`r` is the immediate reward.
    - :math:`\gamma` is the discount factor.
    - :math:`s'` is the next state.
    - :math:`\hat{V}(s') = 0` if the next state is terminal.

    Args:
        feature_dim (int): The dimensionality of the feature representation.
        feature_extractor (Callable[[StateT], np.ndarray]): A function that takes a state
            and returns its feature representation.
        step_size (float, optional): The learning rate for updating the weights. Defaults
            to 0.01.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_extractor: Callable[[StateT], np.ndarray],
        step_size: float = 0.01,
    ) -> None:
        """Initialize the state-value function approximator."""
        self.feature_extractor = feature_extractor
        self.step_size = step_size
        self.weight = np.zeros(feature_dim)

    def get_v(self, state: StateT) -> float:
        """Returns the estimated value of the given state using the linear approximation.

        Args:
            state (StateT): The state for which to estimate the value.

        Returns:
            float: The estimated value of the state.
        """
        value = np.dot(self.weight, self.feature_extractor(state))
        return float(value)

    def update(
        self,
        state: StateT,
        reward: float,
        next_state: StateT,
        discount: float,
        terminal: bool,
    ) -> None:
        r"""Updates the weights of the linear state-value function approximator using
        semi-gradient TD(0).

        Args:
            state (StateT): The current state.
            reward (float): The immediate reward received after taking an action in the
                current state.
            next_state (StateT): The next state after taking an action in the current state.
            discount (float): The discount factor :math:`\gamma`.
            terminal (bool): A boolean indicating whether the next state is terminal.
        """
        V_hat = 0.0 if terminal else self.get_v(next_state)

        features = self.feature_extractor(state)
        target = reward + discount * V_hat
        delta = target - np.dot(self.weight, features)

        self.weight += self.step_size * delta * features


# Problem 6c: Policy Gradient (REINFORCE)
class PolicyGradient[StateT, ActionT](RLAlgorithm[StateT, ActionT]):
    r"""A RL agent that uses the REINFORCE policy gradient algorithm to learn a policy.

    The agent uses a softmax policy parameterized by a weight matrix :attr:`self.theta`
    with shape ``(feature_dim, num_actions)``. Each column corresponds to the weights
    of one action.

    For a state :math:`s`, the policy computes action probabilities as:

    .. math::
        \pi_\theta(a \mid s) =
        \frac{e^{\theta_a^\top\phi(s)}}{\sum_b e^{\theta_b^\top\phi(s)}}

    where :math:`\phi(s)` is the feature vector of state :math:`s`.

    The agent collects transitions during each episode and updates the policy parameters
    at the end of the episode using the REINFORCE algorithm. A running average of past
    episode returns is maintained as a baseline to reduce the variance of policy gradient
    updates.

    .. note::
        Unlike the reward-to-go formulation of REINFORCE in CS234, this implementation
        uses the full discounted episode return for every policy gradient update. It also
        maintains a scalar baseline as a running average of episode returns, initialized
        to the first rollout's utility and updated after each policy update. In contrast,
        REINFORCE can also use a state-dependent baseline, such as an estimated state
        value function.

    Args:
        feature_dim (int): Number of features representing each state.
        feature_extractor (Callable): Function that converts a state into a NumPy feature
            vector of shape ``(feature_dim,)``.
        actions (list[ActionT]): List of available actions.
        discount (float, optional): Discount factor :math:`\gamma` for future rewards.
            Defaults to 0.9.
        time_limit (int, optional): Maximum number of steps per episode. Defaults to 1000.
        step_size (float, optional): Learning rate for updating policy weights. Defaults
            to 1e-5.
        baseline_step_size (float, optional): Learning rate for updating the baseline.
            Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_extractor: Callable[[StateT], np.ndarray],
        actions: list[ActionT],
        discount: float = 0.9,
        time_limit: int = 1000,
        step_size: float = 1e-5,
        baseline_step_size: float = 0.1,
        seed: int | None = None,
    ):
        """Initialize the ``PolicyGradient`` agent with the specified parameters."""
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor

        self.actions = list(dict.fromkeys(actions))
        self.discount = discount
        self.time_limit = time_limit
        self.step_size = step_size
        self.baseline_step_size = baseline_step_size
        self.rng = np.random.default_rng(seed)

        # self.theta has shape `(feature_dim, num_actions)`.
        # self.theta[:, i] holds the weights for the action `i` with index `i`.
        # All zeros gives every action the same probability, so the agent starts
        # out acting uniformly at random.
        self.theta = np.zeros((feature_dim, len(actions)))

        # self.baseline: a running average of the utilities of past rollouts.
        # None until the first rollout ends.
        self.baseline: float | None = None

        # The current rollout: a `(state, action, reward, next_state, terminal)`
        # tuple for each step.
        self.rollout: list[MDPTransition[StateT, ActionT]] = []

    def get_action_probs(self, state: StateT) -> np.ndarray:
        r"""Compute action probabilities using softmax policy.

        The softmax policy is defined as:

        .. math::
            \pi_\theta(a \mid s) =
            \frac{e^{\theta_a^\top\phi(s)}}{\sum_b e^{\theta_b^\top\phi(s)}}

        Args:
            state (StateT): The current environment state.

        Returns:
            Action probabilities of shape ``(num_actions,)``.
        """
        features = self.feature_extractor(state)
        preferences = features @ self.theta
        return softmax(preferences)

    @override
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        r"""Sample an action from the current policy given a state.

        The action is sampled according to :math:`\pi_\theta(a \mid s)`. The :attr:`explore`
        parameter is ignored because the policy is stochastic.

        Args:
            state (StateT): The current environment state.
            explore (bool, optional): Unused. Defaults to ``True``.

        Returns:
            ActionT: The sampled action from :attr:`self.actions`.
        """
        probs = self.get_action_probs(state)
        actions = np.array(self.actions)
        return self.rng.choice(actions, p=probs)

    def log_prob_gradient(self, state: StateT, action: ActionT) -> np.ndarray:
        r"""Compute the gradient of :math:`\log\pi_\theta(a \mid s)`.

        When using linear softmax policy, the gradient of :math:`\log\pi_\theta(a \mid s)`
        can be computed as follows:

        .. math::
            \nabla_\theta \log\pi_\theta(a \mid s) =
            \phi(s)(I_a - \pi_\theta(\cdot \mid s))^\top

        Args:
            state (StateT): Current environment state.
            action (int): Selected action.

        Returns:
            np.ndarray: Gradient with shape ``(feature_dim, num_actions)``.
        """
        features = self.feature_extractor(state)
        probs = self.get_action_probs(state)

        # Gradient for all actions
        grad = -np.outer(features, probs)

        # Add features for the selected action
        idx = self.actions.index(action)
        grad[:, idx] += features

        return grad

    def update_policy(self) -> None:
        r"""Update the policy using transitions collected in the rollout.

        First, compute the discounted return for the rollout:

        .. math::

            U(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t

        Then, compute the policy gradient using REINFORCE with an optional baseline:

        .. math::

            \nabla_\theta J(\theta, \tau) =
            \sum_t (U(\tau) - b) \nabla_\theta \log \pi_\theta(a_t \mid s_t)

        Update the policy parameters using gradient ascent:

        .. math::

            \theta \leftarrow \theta + \alpha_\theta \nabla_\theta J(\theta, \tau)

        If a baseline is used, update it toward the observed return to reduce the variance
        of policy gradient estimates:

        .. math::

            b \leftarrow b + \alpha_b (U(\tau) - b)

        This method is called once after each rollout is completed.

        .. note::

            Following CS221, the baseline is initialized to the utility :math:`U(\tau)`
            of the first rollout. As a result, the policy parameters remain unchanged
            in the first rollout.
        """
        # 1. Compute the discounted return of the entire rollout
        U = 0.0
        for T in reversed(self.rollout):
            U = T.reward + self.discount * U

        # 2. Initialize baseline on the first rollout
        if self.baseline is None:
            self.baseline = U

        # 3. Compute the policy gradient
        gradient = np.zeros_like(self.theta)
        for T in self.rollout:
            log_grad = self.log_prob_gradient(T.state, T.action)
            gradient += (U - self.baseline) * log_grad

        # 4. Update policy parameters
        self.theta += self.step_size * gradient

        # 5. Update baseline
        self.baseline += self.baseline_step_size * (U - self.baseline)

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        r"""Perform one step of the REINFORCE algorithm.

        This function is called by the simulator after each step of the episode. It
        appends the current step's transition to :attr:`self.rollout`, and if the episode
        ends, it calls :meth:`update_policy()` to update the policy parameters theta
        and clears the rollout for the next episode.

        Unlike Q-learning, REINFORCE waits until the rollout is complete before updating
        the policy because it needs the full sequence of rewards to compute returns.

        Args:
            transition (MDPTransition): A tuple containing the current step's transition
                information: state, action, reward, next_state, and terminal flag.
        """
        # Unlike Q-learning, REINFORCE can't update after every step.
        # The weight U(\tau) - b needs the rewards of the whole rollout.
        # So for now we only remember the step.
        self.rollout.append(transition)

        # The rollout ends when the car reaches the flag (terminal is True) or
        # runs out of time.
        # simulate() ends an episode after `time_limit` steps without passing
        # `terminal=True`, so we also count steps.
        if transition.terminal or len(self.rollout) == self.time_limit:
            self.update_policy()  # One gradient ascent step on J
            self.rollout.clear()  # Start a new rollout


# Problem 6f: Bootstrapped Policy Gradient
class BootstrappedPolicyGradient[StateT, ActionT](PolicyGradient[StateT, ActionT]):
    r"""Rollout-based Actor-Critic with TD(0) advantage estimation.

    This agent extends :class:`PolicyGradient` by introducing a learned state-value
    function as a critic. Unlike standard REINFORCE, which uses Monte Carlo returns,
    this algorithm uses bootstrapped TD(0) advantage estimates to update the policy.

    The actor and critic are implemented within the same agent:

    - **Actor:** The policy :math:`\pi_\theta(a \mid s)`, parameterized by :attr:`theta`.
    - **Critic:** The state-value function :math:`V_\phi(s)`, implemented by :attr:`value_function`.

    The critic estimates the one-step TD advantage:

    .. math::

        \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)

    The actor uses this estimate for policy gradient ascent, while the critic learns
    using semi-gradient TD(0). Both components are updated after each completed rollout,
    with the actor updated before the critic.

    Args:
        feature_dim (int): Number of features representing each state.
        feature_extractor (Callable[[StateT], np.ndarray]): Function that converts a state
            into a NumPy feature vector of shape ``(feature_dim,)``.
        actions (list[ActionT]): List of available actions.
        discount (float, optional): Discount factor :math:`\gamma` for future rewards.
            Defaults to 0.9.
        time_limit (int, optional): Maximum number of steps per episode. Defaults to 1000.
        step_size (float, optional): Learning rate for updating policy weights. Defaults
            to 1e-5.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_extractor: Callable[[StateT], np.ndarray],
        actions: list[ActionT],
        discount: float,
        time_limit: int,
        step_size: float = 1e-5,
    ):
        """Initialize the ``BootstrappedPolicyGradient`` agent with the specified parameters."""
        super().__init__(
            feature_dim,
            feature_extractor,
            actions,
            discount=discount,
            time_limit=time_limit,
            step_size=step_size,
        )
        self.value_function = LinearValueFunction(
            feature_dim,
            feature_extractor,
            step_size=step_size,
        )

    @override
    def update_policy(self) -> None:
        r"""Update the actor using bootstrapped TD(0) advantages.

        This method performs the policy gradient update of a rollout-based Actor-Critic
        algorithm. The actor is parameterized by :attr:`theta`, while the critic is
        represented by :attr:`value_function`.

        For each transition in the rollout, compute the one-step TD advantage:

        .. math::

            \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),

        where :math:`V_\phi(s_{t+1}) = 0` for terminal transitions.

        The policy gradient estimate is accumulated over the rollout:

        .. math::

            \hat{g} = \sum_t \delta_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)

        The actor parameters are then updated by gradient ascent:

        .. math::

            \theta \leftarrow \theta + \alpha_\theta \hat{g}

        Unlike standard REINFORCE with a baseline, which uses the Monte Carlo advantage,
        this method uses a bootstrapped TD(0) advantage estimate.

        .. note::

            The critic parameters remain fixed throughout this method. The critic is
            updated afterward by :meth:`step` using semi-gradient TD(0). The actor update
            occurs once per completed rollout rather than after every environment transition.
        """
        gradient = np.zeros_like(self.theta)

        for T in self.rollout:
            if T.terminal:
                next_value = 0.0
            else:
                next_value = self.value_function.get_v(T.next_state)

            advantage = (
                T.reward
                + self.discount * next_value
                - self.value_function.get_v(T.state)
            )
            gradient += advantage * self.log_prob_gradient(T.state, T.action)

        self.theta += self.step_size * gradient

    @override
    def step(self, transition: MDPTransition[StateT, ActionT]) -> None:
        r"""Perform one step of the REINFORCE algorithm.

        This function is called by the simulator after each step of the episode. It
        appends the current step's transition to :attr:`self.rollout`, and if the episode
        ends, it calls :meth:`update_policy()` to update the policy parameters theta, the
        baseline value function, and clears the rollout for the next episode.

        Args:
            transition (MDPTransition): A tuple containing the current step's transition
                information: state, action, reward, next_state, and terminal flag.
        """
        self.rollout.append(transition)

        if transition.terminal or len(self.rollout) == self.time_limit:
            self.update_policy()

            for T in self.rollout:
                self.value_function.update(
                    T.state, T.reward, T.next_state, self.discount, T.terminal
                )

            self.rollout.clear()
