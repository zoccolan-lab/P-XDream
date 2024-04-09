from collections import defaultdict
from os import path
from typing import Any, List

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from zdream.logger import Logger, MutedLogger
from zdream.utils.misc import SEM


def plot_hyperparam(
    hyperparam: str,
    values: List[Any],
    scores: List[List[NDArray]],
    out_dir: str,
    logger: Logger = MutedLogger()
):
    
    COLORS = [
        '#E74C3C', 
        '#F39C12', 
        '#27AE60', 
        '#3498DB', 
        '#A569BD', 
        '#7F8C8D'
    ]
    
    # Compute mean of each population
    means = [[np.mean(iter_score) for iter_score in exp_scores] for exp_scores in scores] 
    
    # Scores across multiple samples grouped by same hyperparameter
    # hyperparameter value: list of scores
    exp_scores = defaultdict(list)

    for param, gen_scores in zip(values, means):
        exp_scores[str(param)].append(gen_scores)
        
    # Computing mean and SEM of scores across samples
    scores_stats = {
        param: tuple(zip(*[
            (np.mean(scores), SEM(scores)) for scores in zip(*gen_scores)
        ])) 
        for param, gen_scores in exp_scores.items()
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
        
    for i, (value, (means, stds)) in enumerate(scores_stats.items()):
        
        # Cast
        means = np.array(means)
        stds  = np.array(stds)
            
        # Get color (circular)
        col = COLORS[i % len(COLORS)]
            
        # Line
        ax.plot(range(len(means)), means, color=col, label=value)

        # IC
        ax.fill_between(
            range(len(stds)),
            means - stds,
            means + stds, 
            color = f'{col}80' # add alpha channel
        )
        
    # Names
    ax.set_xlabel('Generations')
    ax.set_ylabel(f'Score')
    ax.set_title(f'Optimization varying {hyperparam}')
    ax.grid(True)
    ax.legend()
    
    
    out_fp = path.join(out_dir, f'tuning_{hyperparam}.png')
    logger.info(mess=f'Saving plot to {out_fp}')
    fig.savefig(out_fp)