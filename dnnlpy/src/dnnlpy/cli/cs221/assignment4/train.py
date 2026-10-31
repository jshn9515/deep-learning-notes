"""
Trains an agent on the mountain car task and plots its rewards per episode.

    > dnnlpy-cs221-a4-train.exe --agent value-iteration

See the usage string for more details.

    > dnnlpy-cs221-a4-train.exe --help
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import dnnlpy
import dnnlpy.cs221.assignment4 as a4

from .utils import plot_rewards

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv('DNNL_LOG_LEVEL', 'INFO').upper(),
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)


def train_value_iteration(args: argparse.Namespace) -> None:
    logger.info('Starting training for value-iteration agent.')

    for i in range(1, args.num_trials + 1):
        mdp = a4.DiscreteGymMDP(
            'MountainCar-v0',
            bins=20,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            discount=0.999,
            time_limit=1000,
        )
        rl = a4.ModelBasedMonteCarlo(
            mdp.actions,
            mdp.discount,
            mdp.num_states,
            mdp.state_to_index,
            calc_val_iter_every=int(1e5),
            epsilon=0.5,
        )

        try:
            logger.info(f'Starting training for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, num_episodes=1000)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(f'Running evaluation for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, num_episodes=500, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        if args.demo:
            mdp = a4.DiscreteGymMDP(
                'MountainCar-v0',
                bins=20,
                low=[-1.2, -0.07],
                high=[0.6, 0.07],
                discount=0.999,
                time_limit=1000,
                render_mode='human',
            )

            try:
                a4.simulate(
                    mdp, rl, num_episodes=1, train=False, verbose=False, demo=True
                )
            finally:
                mdp.close()

        results = {
            idx: rl.actions[action]
            for idx, action in enumerate(rl.pi_indices)
            if action != -1
        }

        with open(args.path / f'mcvi_weights_trial_{i}.json', 'w') as fp:
            json.dump(results, fp, indent=2, sort_keys=True)
        logger.info(f'Saved weights for trial {i} of value-iteration agent.')

        plot_rewards(train_rewards, eval_rewards, args.path / f'mcvi_trial_{i}.svg')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='dnnlpy-cs221-a4-train.exe',
        description='Train and evaluate various agents on the MountainCar environment.',
        epilog='Example: `dnnlpy-cs221-a4-train.exe --agent tabular`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        metavar='AGENT',
        choices=[
            'value-iteration',
            'tabular',
            'function-approx',
            'constrained',
            'policy-gradient',
        ],
        help='Choose the type of agent to train. Available options are: '
        "1. Model-based value iteration (`'value-iteration'`), "
        "2. Tabular Q-learning (`'tabular'`), "
        "3. Function approximation Q-learning (`'function-approx'`), "
        "4. Constrained Q-learning (`'constrained'`), "
        "5. Policy gradient (`'policy-gradient'`).",
    )
    parser.add_argument(
        '--max_speed',
        type=float,
        default=0.07,
        help='Function-approximation only (5b): Train on CustomMountainCar-v0 environment '
        'with its velocity limit set to `MAX_SPEED`.',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--greedy-eval',
        action='store_true',
        help='Policy-gradient only (6e): Evaluate each trained agent both by sampling '
        'from $\\pi_{\\theta}$ and greedily.',
    )
    group.add_argument(
        '--compare-baseline',
        action='store_true',
        help='Policy-gradient only (6d): Train an agent with the baseline and one without '
        'it (b=0) from the same seed.',
    )
    group.add_argument(
        '--bootstrap',
        action='store_true',
        help='Policy-gradient only (6g): Train tabular Q-learning, function approximation '
        'Q-learning, REINFORCE, and bootstrapped policy gradient from the same seed.',
    )
    parser.add_argument(
        '--num_trials',
        type=int,
        default=3,
        help='The number of trials to run.',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='GymMDP only: Run a demo of the trained agent.',
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=dnnlpy.get_experiment_path() / 'cs221' / 'assignment4',
        help='Specify the directory to save the results. '
        'This will override the default directory, which is `~/dnnlpy/experiments`.',
    )
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.',
    )
    args = parser.parse_args()

    if args.greedy_eval or args.compare_baseline or args.bootstrap:
        if args.agent != 'policy-gradient':
            parser.error(
                'The options `--greedy-eval`, `--compare-baseline`, and `--bootstrap` are '
                'only valid with the `policy-gradient` agent.'
            )

    args.path.mkdir(parents=True, exist_ok=True)
    if args.agent == 'value-iteration':
        train_value_iteration(args)


if __name__ == '__main__':
    main()
