import itertools
import math
from os import path
import os
from typing import Any, Callable, Dict, List, Tuple
import seaborn as sns

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from torchvision.transforms.functional import to_pil_image

from experiment.utils.misc import make_dir
from pxdream.clustering.cluster import Cluster, Clusters
from pxdream.generator import Generator
from pxdream.utils.logger import Logger, SilentLogger
from pxdream.utils.misc import SEM, concatenate_images
from pxdream.utils.types import Codes

from pxdream.utils.misc import SEM


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

def generate_clusters_superstimuli(
    superstimulus: Dict[int, Dict[str, Any]],
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger(),
):
    
    out_superstim_fp = make_dir(path.join(out_dir, 'cluster_superstimuli'))

    for clu_idx, superstim in superstimulus.items():

        best_code = superstim['code']
        best_code_ = np.expand_dims(best_code, axis=0)
        
        best_stim = generator(best_code_)[0]
        best_stim_img = to_pil_image(best_stim)
        
        
        out_fp = path.join(out_superstim_fp, f'clu{clu_idx}_superstimulus.png')
        logger.info(mess=f'Saving superstimulus to {out_fp}')
        best_stim_img.save(out_fp)

def generate_cluster_units_superstimuli(
    superstimulus: Dict[int, Dict[str, Any]],
    clusters: Clusters,
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger(),
):
    
    assert(clusters.obj_tot_count == len(superstimulus))

    out_superstim_fp = make_dir(path.join(out_dir, 'cluster_units_superstimuli'))

    for clu_idx, cluster in enumerate(clusters): # type: ignore

        labels = cluster.labels

        side = int(math.sqrt(len(labels)))
        if side * side < len(labels): side += 1

        codes =  np.stack([superstimulus[i]['code'] for i in labels])

        superstimuli = generator(codes)

        superstimuli_combined = concatenate_images(img_list=list(superstimuli), nrow=side)

        out_fp = path.join(out_superstim_fp, f'clu{clu_idx}_units_superstimuli.png')
        logger.info(mess=f'Saving superstimulus to {out_fp}')
        superstimuli_combined.save(out_fp)

def plot_clusters_superstimuli(
    superstimulus: Dict[int, Dict[str, Any]],
    normalize_fun: Callable[[int], float],
    clusters: Clusters,
    out_dir: str,
    logger: Logger = SilentLogger(),
    **kwargs
):
    
    global HANDLES
    
    # --- MACROS ---

    FIGSIZE = kwargs.get('FIGSIZE', (21, 9))  # Slightly larger figure for better spacing
    PALETTE = kwargs.get('PALETTE', sns.color_palette("crest_r", len(superstimulus)))  # Custom color palette

    FONT = kwargs.get('FONT', 'serif')

    TITLE      = kwargs.get('TITLE', 'Cluster')
    TITLE_ARGS = kwargs.get('TITLE_ARGS', {'fontsize': 20, 'fontfamily': FONT})
    LABEL_ARGS = kwargs.get('LABEL_ARGS', {'fontsize': 16, 'fontfamily': FONT, 'labelpad': 10})
    TICK_ARGS  = kwargs.get('TICK_ARGS', {'labelsize': 14, 'direction': 'out', 'length': 6, 'width': 2})
    GRID_ARGS  = kwargs.get('GRID_ARGS', {'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7})
    LEGEND_ARGS = kwargs.get('LEGEND_ARGS', {'frameon': True, 'fancybox': True, 'framealpha': 0.7, 'loc': 'best', 'prop': {'family': FONT, 'size': 14}})
    MARKER_STYLE = kwargs.get('MARKER_STYLE', {'fmt': '.', 'markersize': 9, 'capsize': 7, 'capthick': 2, 'elinewidth': 2, 'alpha': 0.9})
    LINE_STYLE = kwargs.get('LINE_STYLE', {'linestyle': '--', 'linewidth': 2, 'alpha': 0.9})
    DASHED_LINE_STYLE = kwargs.get('DASHED_LINE_STYLE', {'linestyle': '--', 'linewidth': 2, 'alpha': 0.5})  # Dashed line style
    
    # Prepare figure and axes
    fig_unorm, ax_unorm = plt.subplots(figsize=FIGSIZE)
    fig_norm,  ax_norm  = plt.subplots(figsize=FIGSIZE)
    
    rand_fitness = []
    avg_values_unorm = []
    avg_values_norm = []

    # Plot each cluster
    for clu_idx, superstim in superstimulus.items():

        cluster  = clusters[clu_idx]
        clu_len  = len(cluster)

        if clu_len == 1: continue

        rand_fit = normalize_fun(clu_len)
        rand_fitness.append(rand_fit)
        
        x = clu_idx + 1  # Start x-ticks from 1
        fitness = superstim['fitness']
        
        for ax, norm_factor, avg_values in [(ax_unorm, 1, avg_values_unorm), (ax_norm, rand_fit, avg_values_norm)]:

            fitness_ = [f / norm_factor for f in fitness]

            if ax == ax_norm and clu_len == 1: 
                fitness_ = [1 for f in fitness_]

            avg = np.mean(fitness_)
            std = SEM(fitness_) # type: ignore
            avg_values.append(avg)  # Store the average value for line plotting
            
            # Error Bars
            ax.errorbar(
                x, avg, yerr=std, color=PALETTE[clu_idx % len(PALETTE)], 
                ecolor=PALETTE[clu_idx % len(PALETTE)], **MARKER_STYLE,
                label=f"Cluster fitness"
            )
    
    # Add a dashed line connecting all error bars using the same palette
    for i in range(1, len(avg_values_unorm)):
        ax_unorm.plot([i, i+1], avg_values_unorm[i-1:i+1], color=PALETTE[i % len(PALETTE)], **DASHED_LINE_STYLE)
        ax_norm.plot([i, i+1], avg_values_norm[i-1:i+1], color=PALETTE[i % len(PALETTE)], **DASHED_LINE_STYLE)
    
    # Customize unnormalized plot
    ax_unorm.plot(
        range(1, len(rand_fitness) + 1), rand_fitness, color='grey', label='Random reference', **LINE_STYLE
    )
    ax_unorm.set_xlabel('Cluster Index', **LABEL_ARGS)
    ax_unorm.set_ylabel('Fitness', **LABEL_ARGS)
    ax_unorm.set_title(f'{TITLE} - Cluster Superstimulus Fitness', **TITLE_ARGS)

    handles, _ = ax_unorm.get_legend_handles_labels()
    handles = [handles[len(handles) // 2], handles[0]]

    ax_unorm.legend(handles, ['Cluster fitness', 'Random reference'], **LEGEND_ARGS)
    
    ax_unorm.grid(True, **GRID_ARGS)
    ax_unorm.tick_params(**TICK_ARGS)
    ax_unorm.set_xticks(range(1, len(superstimulus) + 1, len(superstimulus) // 10))  # Ensure x-ticks start from 1
    
    # Customize normalized plot
    ax_norm.set_xlabel('Cluster Index', **LABEL_ARGS)
    ax_norm.set_ylabel('Rand-Normalized Fitness', **LABEL_ARGS)
    ax_norm.set_title(f'{TITLE} - Rand-Normalized Cluster Superstimulus Fitness', **TITLE_ARGS)
    ax_norm.grid(True, **GRID_ARGS)
    ax_norm.tick_params(**TICK_ARGS)
    ax_norm.set_xticks(range(1, len(superstimulus) + 1, len(superstimulus) // 10))  # Ensure x-ticks start from 1

    
    plot_dir = path.join(out_dir, 'plots')
    make_dir(plot_dir)
    
    for fig, name in [
        (fig_unorm, 'cluster_superstimuli_optimization'),
        (fig_norm, 'cluster_superstimuli_optimization_normalized'),
    ]:
        for ext in ['.svg']:
            out_fp = path.join(plot_dir, f'{name}{ext}')
            logger.info(mess=f'Saving plot to {out_fp}')
            fig.savefig(out_fp, bbox_inches='tight', dpi=300)

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
    LEGEND_ARGS = {'frameon': True, 'framealpha': 0.8, 'prop': {'family': FONT, 'size': 14}}
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
        fig.legend(handles, labels, loc='lower center', ncol=3, **LEGEND_ARGS)

        # Add a common title for the entire figure
        fig.suptitle(f'Subset Activations - Cluster {clu_idx}', **TITLE_ARGS)

        # Adjust layout to accommodate the title
        fig.subplots_adjust(top=0.9)

        # Save the figure to the specified output directory
        out_fp = os.path.join(out_dir, f'clu{clu_idx}-{file_name}')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp, dpi=300, bbox_inches='tight')


def plot_dsweighting_clusters_superstimuli(
    superstimuli: Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]],
    out_dir: str,
    logger: Logger = SilentLogger(),
    **kwargs
):
    
    super1, super2 = superstimuli
    assert(len(super1) == len(super2))

    # --- MACROS ---
    FIGSIZE = kwargs.get('FIGSIZE', (21, 9))
    FONT = kwargs.get('FONT', 'serif')

    TITLE             = kwargs.get('TITLE', '')
    TITLE_ARGS        = kwargs.get('TITLE_ARGS',        {'fontsize': 20, 'fontfamily': FONT})
    LABEL_ARGS        = kwargs.get('LABEL_ARGS',        {'fontsize': 16, 'fontfamily': FONT, 'labelpad': 10})
    TICK_ARGS         = kwargs.get('TICK_ARGS',         {'labelsize': 14, 'direction': 'out', 'length': 6, 'width': 2})
    GRID_ARGS         = kwargs.get('GRID_ARGS',         {'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7})
    LEGEND_ARGS       = kwargs.get('LEGEND_ARGS',       {'frameon': True, 'fancybox': True, 'framealpha': 0.7, 'loc': 'best', 'prop': {'family': FONT, 'size': 14}})
    MARKER_STYLE      = kwargs.get('MARKER_STYLE',      {'fmt': '.', 'markersize': 9, 'capsize': 7, 'capthick': 2, 'elinewidth': 2, 'alpha': 0.9})
    DASHED_LINE_STYLE = kwargs.get('DASHED_LINE_STYLE', {'linestyle': '--', 'linewidth': 2, 'alpha': 0.5})

    LABELS   = kwargs.get('LABELS', ['Arithmetic', 'Weighted'])
    PALETTES = kwargs.get('PALETTES', [sns.color_palette("crest_r", len(super1)), sns.color_palette("flare_r", len(super1))])
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    for idx, (superstimulus, palette, label) in enumerate(zip(superstimuli, PALETTES, LABELS)):
        avg_values = []

        for clu_idx, superstim in superstimulus.items():
            x = clu_idx + 1  # Start x-ticks from 1

            fitness = superstim['fitness']
            avg = np.mean(fitness)
            std = SEM(fitness)
            avg_values.append(avg)

            ax.errorbar(
                x, avg, yerr=std, color=palette[clu_idx % len(palette)], 
                ecolor=palette[clu_idx % len(palette)], label=f"{label} - Cluster {clu_idx + 1}", **MARKER_STYLE
            )
        
        # Connect points with dashed line
        for i in range(1, len(avg_values)):
            ax.plot([i, i+1], avg_values[i-1:i+1], color=palette[i % len(palette)], **DASHED_LINE_STYLE)
    
    # Customize plot
    ax.set_xlabel('Cluster Index', **LABEL_ARGS)
    ax.set_ylabel('Fitness', **LABEL_ARGS)
    ax.set_title(f'{TITLE} - Dominant Set Clustering, Fitness Weighting Comparison', **TITLE_ARGS)
    
    ax.grid(True, **GRID_ARGS)
    ax.tick_params(**TICK_ARGS)
    ax.set_xticks(range(1, len(superstimuli[0]) + 1, len(superstimuli[0]) // 10))  # Ensure x-ticks start from 1
    
    # Add legends only once, combining them
    handles, _ = ax.get_legend_handles_labels()
    quarter = len(handles) // 4
    handles = [handles[quarter], handles[quarter*3]]

    ax.legend(handles, LABELS, **LEGEND_ARGS)
    
    plot_dir = make_dir(path.join(out_dir, 'plots'))
    
    for ext in ['.svg']:
        out_fp = path.join(plot_dir, f'comparative_ds_weighting{ext}')
        logger.info(mess=f'Saving comparative plot to {out_fp}')
        fig.savefig(out_fp, bbox_inches='tight', dpi=300)