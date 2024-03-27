from collections import defaultdict
from os import path
from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from zdream.logger import Logger, MutedLogger
from zdream.utils.misc import SEM


def plot_scr(
    cluster_idx: List[int],
    weighted: List[bool],
    scores: List[List[NDArray]],
    out_dir: str,
    logger: Logger = MutedLogger()
):
    
    # Line and background
    COLORS = {
        'arithmetic' :  ('#229954', '#22995480'),
        'weighted'   :  ('#A569BD', '#A569BD80')
    }
    
    
    # Compute means and max of population for each generation
    means, maxs = (
        [
            [op(iter_score) for iter_score in exp_score] 
            for exp_score in scores
        ] for op in [np.mean, np.max]
    )
    
    # Organize data for computation
    # cluster_id: statistic (mean or max) : score type (weighted or arithmetic): scores
    clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, weight, scores in zip(cluster_idx, weighted, zip(means, maxs)): # type: ignore
        mean_, max_ = scores
        w = "weighted" if weight else 'arithmetic'
        clusters[idx]['mean'][w].append(mean_)
        clusters[idx]['max' ][w].append(max_)
    
    # Compute mean and standard deviation across samples
    clusters = {
        idx: {
            stat: {
                scr_type: list(
                    zip(*[(np.mean(iter_samples) ,SEM(iter_samples)) 
                    for iter_samples in list(zip(*generations))])
                )
                for scr_type, generations in scores.items()
            }
            for stat, scores in cluster_info.items()
        }
        for idx, cluster_info in clusters.items()
    }
    
    # Plot
    for clu_idx, cluster_info in clusters.items():
    
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        
        for i, (stat, genreations) in enumerate(cluster_info.items()):
            
            for scr_type, scores in genreations.items():
                
                means, stds = scores
                
                means = np.array(means)
                stds  = np.array(stds)
                
                col1, col2 = COLORS[scr_type]
                
                # Line
                axes[i].plot(range(len(means)), means, color=col1, label=scr_type)

                # IC
                axes[i].fill_between(
                    range(len(stds)),
                    means - stds,
                    means + stds, 
                    color = col2
                )
            
            axes[i].set_xlabel('Generations')
            axes[i].set_ylabel(f'{stat.capitalize()} activation')
            axes[i].set_title('')
            axes[i].grid(True)
            axes[i].legend()
        
        fig.suptitle(f'Cluster {clu_idx}')
        
        out_fp = path.join(out_dir, f'cluster_{clu_idx}.png')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)