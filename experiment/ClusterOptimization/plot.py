import itertools
import math
from os import path
from typing import Any, Callable, Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from torchvision.transforms.functional import to_pil_image

from experiment.utils.misc import make_dir
from zdream.clustering.cluster import Clusters
from zdream.generator import Generator
from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.misc import SEM, concatenate_images
from zdream.utils.types import Codes

from os import path
from typing import Dict, List


def plot_ds_weigthed_score(
    scores  : Dict[str, Dict[str, List[NDArray]]],
    out_dir : str,
    logger  : Logger = SilentLogger()
):
    
    # Line and background
    COLORS = {
        'arithmetic' :  ('#229954', '#22995480'),
        'weighted'   :  ('#A569BD', '#A569BD80')
    }
    FIGSIZE = (8, 8)
    
    # Compute mean and std of scored
    
    # Plot
    for clu_idx, activations in scores.items():
    
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        for weighted, generations in activations.items():
            
            scr_type = 'weighted' if weighted else 'arithmetic'
            
            in_sample_gen = [[gen[i] for gen in generations] for i in range(len(generations[0]))]

            means = np.array([float(np.mean(gen)) for gen in in_sample_gen])
            stds  = np.array([float(    SEM(gen)) for gen in in_sample_gen])
            
            col1, col2 = COLORS[scr_type]
            
            # Line
            ax.plot(range(len(means)), means, color=col1, label=scr_type)

            # IC
            ax.fill_between(
                range(len(stds)),
                means - stds,
                means + stds, 
                color = col2
            )
        
        ax.set_xlabel('Generations')
        ax.set_ylabel(f'Fitness')
        ax.set_title('Fitness across generation with arithmetic and weighted score')
        ax.grid(True)
        ax.legend()
        
        out_fp = path.join(out_dir, f'clu_{clu_idx}_weighted_ds_fitness.png')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)

def plot_clusters_superstimuli(
    superstimulus: Dict[int, Dict[str, Any]],
    normalize_fun: Callable[[int], float],
    clusters: Clusters,
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    FIGSIZE    = (16, 8)
    COLOR_CLU  = '#DC7633'
    COLOR_RND  = '#3498DB'
    EDGE_COLOR = 'black'

    # Save the superstimuli
    for clu_idx, superstim in superstimulus.items():
        
        best_code = superstim['code']
        best_code_ = np.expand_dims(best_code, axis=0)
        
        best_stim = generator(best_code_)[0]
        best_stim_img = to_pil_image(best_stim)
        
        out_superstim_fp = path.join(out_dir, 'superstimuli')
        make_dir(out_superstim_fp)
        
        out_fp = path.join(out_superstim_fp, f'clu{clu_idx}_superstimulus.png')
        best_stim_img.save(out_fp)
    
    
    errorbars,      ax1      = plt.subplots(figsize=FIGSIZE)
    boxplots,       ax2      = plt.subplots(figsize=FIGSIZE)
    errorbars_norm, ax1_norm = plt.subplots(figsize=FIGSIZE)
    boxplots_norm,  ax2_norm = plt.subplots(figsize=FIGSIZE)
    
    rand_fitness = []
    
    # Plot
    for clu_idx, superstim in superstimulus.items():
        
        cluster  = clusters[clu_idx]  # type: ignore
        clu_len  = len(cluster)
        rand_fit = normalize_fun(clu_len)
        rand_fitness.append(rand_fit)
        
        x       = clu_idx
        fitness = superstim['fitness']
        
        for axes, norm_factor in zip([(ax1, ax2), (ax1_norm, ax2_norm)], (1., rand_fit)):
        
            ax_err, ax_box = axes
        
            fitness_ = [f / norm_factor for f in fitness]
            
            avg = np.mean(fitness_)
            std = SEM(fitness_)
            
            # Error Bars
            ax_err.errorbar(x, avg, yerr=std, color=COLOR_CLU, ecolor=COLOR_CLU, fmt='o', markersize=8, capsize=10)
            # Boxplot
            boxplot = ax_box.boxplot(fitness_, positions=[x], patch_artist=True, boxprops=dict(facecolor=COLOR_CLU, color=EDGE_COLOR))
    
    # Adding labels outside the loop to avoid multiple legend entries
    ax1.errorbar([], [], color=COLOR_CLU, ecolor=COLOR_CLU, fmt='o', markersize=8, capsize=10, label='Cluster fitness')
    ax2.boxplot([[]], patch_artist=True, boxprops=dict(facecolor=COLOR_CLU, color=EDGE_COLOR)).get('boxes')[0].set_label('Cluster fitness')
    
    # Customize axes    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Fitness')
        ax.set_title('Cluster supertimulus fitness')
        ax.set_xticks(list(superstimulus.keys()))
        ax.plot(list(superstimulus.keys()), rand_fitness, color=COLOR_RND, label='Random reference', linestyle='--')
        ax.legend()
    
    for ax in [ax1_norm, ax2_norm]:    
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Rand-Normalized Fitness')
        ax.set_title('Cluster supertimulus fitness normalized on random fitness')
        ax.set_xticks(list(superstimulus.keys()))
        
    # Save figures
    plot_dir = path.join(out_dir, 'plots')
    make_dir(plot_dir)
    for fig, name in [
        #(errorbars,      'errorbars'),
        (boxplots,       'boxplots'),
        #(errorbars_norm, 'errorbars_norm'),
        (boxplots_norm,  'boxplots_norm'),
    ]:
        out_fp = path.join(plot_dir, f'{name}.svg')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)

def plot_activations(
    activations: Dict[str, NDArray],
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    FIG_SIZE = (8, 7)
    
    COLORS   = {
        'Cluster optimized'      : '#0013D6', 
        'External non optimized' : '#E65c00',
        'Cluster non optimized'  : '#36A900'
    }
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for name, act in activations.items():

        color = COLORS[name]

        means = np.mean(act, axis=1)
        stds  =     SEM(act, axis=1)

        ax.plot(range(len(means)), means, color=color, label=name)

        # IC
        ax.fill_between(
            range(len(stds)),
            means - stds,
            means + stds, 
            color = f'{color}80' # add alpha channel
        )
            
        # Names
        ax.set_xlabel('Generations')
        ax.set_ylabel('Avg. Activations')
        ax.set_title(f'Componentes aggregated activations')
        ax.grid(True)
        ax.legend()

    out_fp = path.join(out_dir, f'subset_optimization.png')
    logger.info(mess=f'Saving plot to {out_fp}')
    fig.savefig(out_fp)

import os
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
import itertools
from numpy.typing import NDArray
from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.misc import SEM
import seaborn as sns

def plot_subset_activations(
    activations: Dict[int, Dict[str, Dict[int, Dict[str, List[float]]]]],
    out_dir: str,
    file_name: str = 'subset_activations.svg',
    logger: Logger = SilentLogger()
):
    # Styling parameters for plots
    PALETTE = 'Dark2'
    FONT = 'serif'
    TICK_SIZE = 10  # Font size for tick labels
    IC_ALPHA = 0.3  # Transparency for confidence intervals
    TITLE_ARGS = {'fontsize': 22, 'fontfamily': FONT}
    SUBTITLE_ARGS = {'fontsize': 14, 'fontfamily': FONT}
    LABEL_ARGS = {'fontsize': 16, 'fontfamily': FONT}
    LEGEND_ARGS = {'fontsize': 14, 'frameon': True, 'framealpha': 0.8}
    GRID_ARGS = {'color': '#333333', 'linestyle': '--', 'alpha': 0.3}
    BOUND_OFFSET = (-0.5, 0.5)  # Offset for y-axis limits
    FIG_SIZE_X, FIG_SIZE_Y = (5, 5)  # Figure size parameters

    # Define color palette using Seaborn
    palette = sns.color_palette(PALETTE)
    COLORS = {
        'Cluster optimized': palette[0],
        'Cluster non optimized': palette[1],
        'External non optimized': palette[2],
    }

    # Determine figure size based on the number of subplots
    first_activation = activations[list(activations.keys())[0]]
    first_scr_type = first_activation[list(first_activation.keys())[0]]
    first_k = first_scr_type[list(first_scr_type.keys())[0]]

    NCOL = len(first_k)
    NROW = 2
    FIG_SIZE = (FIG_SIZE_X * NCOL, FIG_SIZE_Y * NROW)

    for clu_idx, topbot_activations in activations.items():
        # Create a new figure and axes for each cluster
        fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, figsize=FIG_SIZE)
        handles = []
        labels = []
        BOUNDS = (0, 0)  # Initialize bounds for y-axis limits

        for scr_type, activs in topbot_activations.items():
            row = 0 if scr_type == 'subset_top' else 1
            title_top_bot = 'Top' if row == 0 else 'Bot'

            for col, (k, act_types) in enumerate(activs.items()):
                for act_name, a in act_types.items():
                    color = COLORS[act_name]

                    # Calculate means and standard errors
                    means = np.mean(a, axis=1)
                    stds = SEM(a, axis=1)

                    # Update bounds for y-axis limits
                    BOUNDS = (
                        min(BOUNDS[0], np.min(means - stds)),
                        max(BOUNDS[1], np.max(means + stds))
                    )

                    ax = axes[row, col]
                    line, = ax.plot(range(len(means)), means, color=color, label=act_name, linewidth=2)

                    # Collect handles and labels for the common legend
                    if act_name not in labels:
                        handles.append(line)
                        labels.append(act_name)

                    # Plot confidence interval (IC) with transparency
                    ax.fill_between(
                        range(len(stds)),
                        means - stds,
                        means + stds,
                        color=color,
                        alpha=IC_ALPHA
                    )

                    # Set titles based on column index
                    title_k = {
                        0: "1 Optimization",
                        1: f"$\sqrt{{N}}$ Optimization ({k} units)",
                        2: f"N/2 Optimization ({k} units)"
                    }.get(col, f"Unknown Optimization ({k} units)")

                    # Set subplot title
                    ax.set_title(f'{title_top_bot}-{title_k}', **SUBTITLE_ARGS)

                    # Customize tick parameters
                    ax.tick_params(axis='x', labelsize=TICK_SIZE)
                    ax.tick_params(axis='y', labelsize=TICK_SIZE)

                    # Apply grid style
                    ax.grid(True, **GRID_ARGS)

                    # Hide ticks and labels based on row and column
                    if row != NROW - 1:
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', length=0)
                    if col != 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', length=0)

                    # Set x-axis and y-axis labels
                    if row == NROW - 1:
                        ax.set_xlabel('Generations', **LABEL_ARGS)
                    if col == 0:
                        ax.set_ylabel('Fitness', **LABEL_ARGS)

                    # Customize spines based on position
                    ax.spines['bottom'].set_visible(row == 1)
                    ax.spines['top'].set_visible(row == 0)
                    ax.spines['left'].set_visible(col == 0)
                    ax.spines['right'].set_visible(col == NCOL - 1)

        # Adjust y-axis limits across all subplots
        BOUNDS = (BOUNDS[0] + BOUND_OFFSET[0], BOUNDS[1] + BOUND_OFFSET[1])
        for i, j in itertools.product(range(NROW), range(NCOL)):
            axes[i][j].set_ylim(BOUNDS)

        # Add a common legend to the figure
        fig.legend(handles, labels, loc='lower center', ncol=3, **LEGEND_ARGS, prop={'family': FONT})

        # Add a common title for the entire figure
        fig.suptitle(f'Subset Activations - Cluster {clu_idx}', **TITLE_ARGS)

        # Adjust layout to accommodate the title
        fig.subplots_adjust(top=0.9)

        # Save the figure to the specified output directory
        out_fp = os.path.join(out_dir, f'clu{clu_idx}-{file_name}')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp, dpi=300, bbox_inches='tight')
