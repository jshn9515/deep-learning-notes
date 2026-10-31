import os

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'moving_average',
    'plot_rewards',
]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Returns the moving average of the input array with the given window size."""
    cum_sum = np.cumsum(x)
    mov_avg = (cum_sum[window:] - cum_sum[:-window]) / window
    return mov_avg


def plot_rewards(
    train_rewards: np.ndarray,
    eval_rewards: np.ndarray,
    filename: str | os.PathLike[str] | None = None,
    show: bool = True,
    greedy_eval_rewards: np.ndarray | None = None,
) -> None:
    """Plots the rewards of training and evaluation episodes, with moving averages.

    If ``greedy_eval_rewards`` is not None, also plots those evaluation episodes (orange)
    on top of the green ones.
    """
    window = 30
    train_mov_avg = moving_average(train_rewards, window)
    eval_mov_avg = moving_average(eval_rewards, window)
    t_len = len(train_rewards)
    e_len = len(eval_rewards)

    plt.rc('font', size=12)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        range(t_len),
        train_rewards,
        s=8,
        c='tab:blue',
        linewidth=0,
        alpha=0.5,
        label='train rewards',
    )
    ax.plot(
        range(window // 2, t_len - window // 2),
        train_mov_avg,
        linewidth=2,
        color='blue',
        label='train moving average',
    )
    ax.scatter(
        range(t_len, t_len + e_len),
        eval_rewards,
        s=8,
        c='tab:green',
        linewidth=0,
        alpha=0.5,
        label='eval rewards (sampling)',
    )
    ax.plot(
        range(t_len + window // 2, t_len + e_len - window // 2),
        eval_mov_avg,
        color='darkgreen',
        linewidth=2,
        label='eval moving average (sampling)',
    )

    if greedy_eval_rewards is not None:
        ax.scatter(
            range(t_len, t_len + len(greedy_eval_rewards)),
            greedy_eval_rewards,
            s=8,
            c='tab:orange',
            alpha=0.5,
            linewidth=0,
            label='eval rewards (greedy)',
        )
        ax.plot(
            range(
                t_len + window // 2,
                t_len + len(greedy_eval_rewards) - window // 2,
            ),
            moving_average(greedy_eval_rewards, window),
            color='darkorange',
            linewidth=2,
            label='eval moving average (greedy)',
        )

    ax.legend(loc='lower right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Discounted Reward in Episode')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
