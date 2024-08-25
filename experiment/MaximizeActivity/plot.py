import math
from os import path
import os
import re
from typing import Any, Dict, Literal, List, Tuple, cast

import PIL
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from pandas import DataFrame
from matplotlib import cm, pyplot as plt
from matplotlib.axes import Axes
from torchvision.transforms.functional import to_pil_image

from analysis.utils.misc import load_imagenet
from zdream.generator import DeePSiMGenerator, DeePSiMVariant, Generator, DeePSiMVariant
from zdream.utils.misc import default, overwrite_dict
from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.dataset import ExperimentDataset
from zdream.utils.misc import SEM, default
from zdream.utils.types import Codes, Scores, Stimuli

# --- DEFAULT PLOTTING PARAMETERS ----

_Shapes = Literal['square', 'rect_tall', 'rect_wide']
_ax_selection = Literal['x', 'y', 'xy']

def _get_appropriate_fontsz(xlabels: List[str], figure_width: float | int | None = None) -> float:
    '''
    Dynamically compute appropriate fontsize for xticks 
    based on the number of labels and figure width.
    
    :param labels: Labels in the abscissa ax.
    :type labels: List[str]
    :param figure_width: Figure width in inches. If not specified using default figure width.
    :type figure_width: float | int | None
    
    :return: Fontsize for labels ticks in abscissa ax.
    :rtype: float
    '''
    
    # Compute longest label length
    max_length = max(len(label) for label in xlabels)
    
    # In case where figure_width not specified, use default figsize
    figure_width_: int | float = default(figure_width, plt.rcParams['figure.figsize'][0])
        
    # Compute the appropriate fontsize to avoid overlaps 
    # NOTE: Ratio factor 0.0085 was found empirically
    fontsz = figure_width_ / len(xlabels) / max_length / 0.0085 
    
    return fontsz


def subplot_same_lims(axes: NDArray, sel_axs:_ax_selection = 'xy'):
    """set xlim, ylim or both to be the same in all subplots

    :param axes: array of Axes objects (one per subplot)
    :type axes: NDArray
    :param sel_axs: axes you want to set same bounds, defaults to 'xy'
    :type sel_axs: _ax_selection, optional
    """
    #lims is a array containing the lims for x and y axes of all subplots
    lims = np.array([[ax.get_xlim(), ax.get_ylim()] for ax in axes.flatten()])
    #get the extremes for x and y axis respectively among subplots
    xlim = [np.min(lims[:,0,:]), np.max(lims[:,0,:])] 
    ylim = [np.min(lims[:,1,:]), np.max(lims[:,1,:])]
    #for every Axis obj (i.e. for every subplot) set the extremes among all subplots
    #depending on the sel_axs indicated (either x, y or both (xy))
    #NOTE: we could also improve this function by allowing the user to change the
    #limits just for a subgroup of subplots
    for ax in axes.flatten():
        ax.set_xlim(xlim) if 'x' in sel_axs else None
        ax.set_ylim(ylim) if 'y' in sel_axs else None   
        
        
def set_default_matplotlib_params(
    l_side  : float     = 15,
    shape   : _Shapes   = 'square', 
    xlabels : List[str] = []
) -> Dict[str, Any]:
    '''
    Set default Matplotlib parameters.
    
    :param l_side: Length of the lower side of the figure, default to 15.
    :type l_side: float
    :param shape: Figure shape, default is `square`.
    :type shape: _Shapes
    :param xlabels: Labels lists for the abscissa ax.
    :type xlabels: list[str]
    
    :return: Default graphic parameters.
    :rtype: Dict[str, Any]
    '''
    
    # Set other dimension sides based on lower side dimension and shape
    match shape:
        case 'square':    other_side = l_side
        case 'rect_wide': other_side = l_side * (2/3)
        case 'rect_tall': other_side = l_side * (3/2)
        case _: raise TypeError(f'Invalid Shape. Use one of {_Shapes}')
    
    # Default params
    fontsz           = 35
    standard_lw      = 4
    box_lw           = 3
    box_c            = 'black'
    subplot_distance = 0.3
    axes_lw          = 3
    tick_length      = 6
    
    # If labels are given compute appropriate fontsz is the minimum
    # between the appropriate fontsz and the default fontsz
    if xlabels:
        fontsz =  min(
            _get_appropriate_fontsz(xlabels=xlabels, figure_width=l_side), 
            fontsz
        )
    
    # Set new params
    params = {
        'figure.figsize'                : (l_side, other_side),
        'font.size'                     : fontsz,
        'axes.labelsize'                : fontsz,
        'axes.titlesize'                : fontsz,
        'xtick.labelsize'               : fontsz,
        'ytick.labelsize'               : fontsz,
        'legend.fontsize'               : fontsz,
        'axes.grid'                     : False,
        'grid.alpha'                    : 0.4,
        'lines.linewidth'               : standard_lw,
        'lines.linestyle'               : '-',
        'lines.markersize'              : 20,
        'xtick.major.pad'               : 5,
        'ytick.major.pad'               : 5,
        'errorbar.capsize'              : standard_lw,
        'boxplot.boxprops.linewidth'    : box_lw,
        'boxplot.boxprops.color'        : box_c,
        'boxplot.whiskerprops.linewidth': box_lw,
        'boxplot.whiskerprops.color'    : box_c,
        'boxplot.medianprops.linewidth' : 4,
        'boxplot.medianprops.color'     : 'red',
        'boxplot.capprops.linewidth'    : box_lw,
        'boxplot.capprops.color'        : box_c,
        'axes.spines.top'               : False,
        'axes.spines.right'             : False,
        'axes.linewidth'                : axes_lw,
        'xtick.major.width'             : axes_lw,
        'ytick.major.width'             : axes_lw,
        'xtick.major.size'              : tick_length,
        'ytick.major.size'              : tick_length,
        'figure.subplot.hspace'         : subplot_distance,
        'figure.subplot.wspace'         : subplot_distance
    }
    
    plt.rcParams.update(params)

    return params

# --- STYLING --- 

def customize_axes_bounds(ax: Axes):
    '''
    Modify axes bounds based on the data type.

    For numeric data, the bounds are set using the values of the second and second-to-last ticks.
    For categorical data, the bounds are set using the indices of the first and last ticks.

    :param ax: Axes object to be modified.
    :type ax: Axes
    '''
    
    # Get tick labels
    tick_positions_x = ax.get_xticklabels()
    tick_positions_y = ax.get_yticklabels()
    
    # Compute new upper and lower bounds for both axes
    t1x = tick_positions_x[ 1].get_text()
    t2x = tick_positions_x[-2].get_text()
    t1y = tick_positions_y[ 1].get_text()
    t2y = tick_positions_y[-2].get_text()
    
    # Set new axes bound according to labels type (i.e. numeric or categorical)
    for side, (low_b, up_b) in zip(['left', 'bottom'], ((t1y, t2y), (t1x, t2x))):
        
        # Case 1: Numeric - Set first and last thicks bounds as floats
        if low_b.replace('−', '').replace('.', '').isdigit():
            low_b = float(low_b.replace('−','-'))
            up_b  = float( up_b.replace('−','-'))
            
            # Remove ticks outside bounds
            ticks = tick_positions_y if side == 'left' else tick_positions_x
            ticks = [float(tick.get_text().replace('−','-')) for tick in ticks 
                     if low_b <= float(tick.get_text().replace('−','-')) <= up_b]
            
            ax.set_yticks(ticks) if side == 'left' else ax.set_xticks(ticks)
            
        # Case 2: Categorical - Set first and last thicks bounds as index
        else:
            ticks = tick_positions_y if side == 'left' else tick_positions_x
            low_b = 0
            up_b  = len(ticks)-1
        
        ax.spines[side].set_bounds(low_b, up_b)
        

def plot_scores(
    scores: Tuple[NDArray, NDArray],
    stats:  Tuple[Dict[str, Any], Dict[str, Any]],
    style: Dict[str, Dict[str, str]] = {
        'gen': {'lbl': 'Synthetic', 'col': 'k'},
        'nat': {'lbl':   'Natural', 'col': 'g'}
    },
    num_bins: int  = 25, 
    out_dir: str | None = None,
    logger: Logger | None = None
):
    '''
    Plot two views of scores trend through optimization steps.
    
    1. The first plot displays the score of the best stimulus per optimization step (left) 
       and the average stimuli score pm SEM per iteration (right), for both natural and synthetic images. 

    2. The second plot displays the histograms of scores for both natural and synthetic images.
    
    :param scores: Scores history of synthetic and natural images.
    :type scores: Tuple[NDArray, NDArray]
    :param stats: Collection of statistics for synthetic and natural images through optimization steps.
    :type stats: Tuple[Dict[str, Any], Dict[str, Any]]
    :param style: Style dictionary setting labels and colors of synthetic and natural images.
                    They default to black for synthetic and to green for natural.
    :type style: Dict[str, Dict[str, str]]
    :param num_bins: Number of bins to display in the histogram, defaults to 25.
    :type num_bins: int
    :param out_dir: Directory where to save output plots, default is None indicating not to save.
    :type out_dir: str | None
    :param logger: Logger to log information relative to plot saving paths.
                   Defaults to None indicating no logging.
    :type logger: Logger | None
    '''

    # Preprocessing input
    scores_gen, scores_nat = scores
    stats_gen,  stats_nat  = stats

    logger = default(logger, SilentLogger())

    # Plot nat
    use_nat = len(scores_nat) != 0 or stats_nat
    
    # Retrieve default parameters and retrieve `alpha` parameter
    def_params = set_default_matplotlib_params(shape='rect_wide', l_side = 30)
    
    # PLOT 1. BEST and AVG SCORE TREND
    fig_trend, ax = plt.subplots(1, 2)
    
    # Define label and colors for both generated and natural images
    lbl_gen = style['gen']['lbl']; col_gen = style['gen']['col']
    if use_nat:
        lbl_nat = style['nat']['lbl']; col_nat = style['nat']['col']

    # We replicate the same reasoning for the two subplots by accessing 
    # the different key of the score dictionary whether referring to max or mean.
    for i, k in enumerate(['best_gens', 'mean_gens']):
        
        # Lineplot of both synthetic and natural scores
        ax[i].plot(stats_gen[k], label=lbl_gen, color=col_gen)
        if use_nat:
            ax[i].plot(stats_nat[k], label=lbl_nat, color=col_nat)
        
        # When plotting mean values, add SEM shading
        if k =='mean_gens':
            for stat, col in zip(
                [stats_gen, stats_nat] if use_nat else [stats_gen], 
                [  col_gen,   col_nat] if use_nat else [  col_gen]
            ):
                ax[i].fill_between(
                    range(len(stat[k])),
                    stat[k] - stat['sem_gens'],
                    stat[k] + stat['sem_gens'], 
                    color=col, alpha=def_params['grid.alpha']
                )
                
        # Names
        ax[i].set_xlabel('Generation cycles')
        ax[i].set_ylabel('Target scores')
        ax[i].set_title(k.split('_')[0].capitalize())
        ax[i].legend()
        # customize_axes_bounds(ax[i])

    # Save or display  
    if out_dir:
        out_fp = path.join(out_dir, 'scores_trend.png')
        logger.info(f'Saving score trend plot to {out_fp}')
        fig_trend.savefig(out_fp, bbox_inches="tight")
    
    # PLOT 2. SCORES HISTOGRAM 
    fig_hist, ax = plt.subplots(1) 
        
    # Compute min and max values
    data_min = min(scores_nat.min(), scores_gen.min()) if use_nat else scores_gen.min()
    data_max = max(scores_nat.max(), scores_gen.max()) if use_nat else scores_gen.max()

    # Create histograms for both synthetic and natural with the same range and bins
    if use_nat:
        hnat = plt.hist(
            scores_nat.flatten(),
            bins=num_bins,
            range=(data_min, data_max),
            alpha=1, 
            label=lbl_nat,
            density=True,
            edgecolor=col_nat, 
            linewidth=3
        )
    
    hgen = plt.hist(
        scores_gen.flatten(),
        bins=num_bins,
        range=(data_min, data_max),
        density=True,
        color = col_gen,
        edgecolor=col_gen
    )
    
    # For generated images set alpha as a function of the generation step.
    # NOTE: Since alpha influences also edge transparency, 
    #       histogram needs to be separated into two parts for edges and filling
    hgen_edg = plt.hist(
        scores_gen.flatten(), 
        bins=num_bins, 
        range=(data_min, data_max), 
        label= lbl_gen,
        density=True,
        edgecolor=col_gen,
        linewidth=3
    )
    
    # Set transparent columns for natural images and edges of synthetic ones
    for bins in [hnat, hgen_edg] if use_nat else [hgen_edg]:
        for bin in bins[-1]: # type: ignore
            bin.set_facecolor('none')
    
    # Retrieve number of iterations
    n_gens = scores_gen.shape[0]
    
    # Set alpha value of each column of the hgen histogram.
    # Iterate over each column, taking its upper bound and its histogram bin
    for up_bound, bin in zip(hgen[1][1:], hgen[-1]): # type: ignore
        
        # Compute the probability for each generation of being
        # less than the upper bound of the bin of interest
        is_less = np.mean(scores_gen < up_bound, axis = 1)
        
        # Weight the alpha with the average of these probabilities.
        # Weights are the indexes of the generations. The later the generation, 
        # the higher its weight (i.e. the stronger the shade of black).
        # TODO Is weighting too steep in this way?
        
        alpha_col = np.sum(is_less*range(n_gens)) / np.sum(range(n_gens))
        bin.set_alpha(alpha_col)
        
    # Plot names
    plt.xlabel('Target score')
    plt.ylabel('Prob. density')
    plt.legend()
    # customize_axes_bounds(ax)
    
    # Save or display  
    if out_dir:
        out_fp = path.join(out_dir, 'scores_histo.png')
        logger.info(f'Saving score histogram plot to {out_fp}')
        fig_hist.savefig(out_fp, bbox_inches="tight")


def plot_scores_by_label(
    scores: Tuple[NDArray, NDArray], 
    lbls: List[int], 
    dataset: ExperimentDataset,
    k: int = 3, 
    gens_window: int = 5,
    out_dir: str | None = None,
    logger: Logger | None = None
):
    '''
    Plot illustrating the average pm SEM scores of the top-k and bottom-k categories of natural images.
    Moreover it also shows the average pm SEM scores of synthetic images within the first and last
    iteration of a considered generation window.
    
    :param scores: Scores history of synthetic and natural images.
    :type scores: Tuple[NDArray, NDArray]
    :param lbls: List of the labels seen during optimization.
    :type lbls: List[int]
    :param dataset: Dataset of natural images allowing for id to category mapping.
    :type dataset: MiniImageNet
    :param k: Number of natural images classes to consider for the top-k and bottom-k.
              Default is 3
    :type k: int
    :param gens_window: Generations window to consider scores at the beginning and end 
                        of the optimization. Default is 5.
    :type n_gens_considered: int
    :param out_dir: Directory where to save output plots, default is None indicating not to save.
    :type out_dir: str | None
    :param logger: Logger to log information relative to plot saving paths.
                   Defaults to None indicating no logging.
    :type logger: Logger | None
    '''

    # Preprocessing input
    scores_gen, scores_nat = scores 

    logger = default(logger, SilentLogger())
    
    # Cast scores and labels as arrays.
    # NOTE: `nat_scores` are flattened to be more easily 
    #        indexed by the `nat_lbls` vector
    gen_scores = scores_gen
    nat_scores = scores_nat.flatten()
    
    # Convert class indexes to labels
    class_to_lbl = np.vectorize(lambda x: dataset.class_to_lbl(lbl=x))
    nat_lbls = class_to_lbl(lbls)
    
    # Get the unique labels present in the dataset
    unique_lbls, _ = np.unique(nat_lbls, return_counts=True)
    
    # Save score statistics of interest (mean, SEM and max) for each of the labels
    lbl_acts = {}
    for lb in unique_lbls:
        lb_scores = nat_scores[np.where(nat_lbls == lb)[0]]
        lbl_acts[lb] = (np.mean(lb_scores), SEM(lb_scores), np.amax(lb_scores))
    
    # Sort scores by their mean
    # NOTE: x[1][0] first takes values of the dictionary 
    #       and then uses the mean as criterion for sorting
    best_labels = sorted(lbl_acts.items(), key=lambda x: x[1][0])
    
    # Extracting top-k and bottom-k categories
    # NOTE: Scores are in ascending order    
    top_categories = best_labels[-k:]
    bot_categories = best_labels[:k]
    
    # Unpacking top and bottom categories for plotting
    top_labels, top_values = zip(*top_categories)
    bot_labels, bot_values = zip(*bot_categories)
    
    # Define the mean and standard deviation of early and late generations
    gen_dict = {
        'Early': (
            np.mean(gen_scores[:gens_window,:]), 
            SEM    (gen_scores[:gens_window,:].flatten())
        ),
        'Late': (
            np.mean(gen_scores[-gens_window:,:]), 
            SEM    (gen_scores[-gens_window:,:].flatten())
        )}
    
    # Plot using default configurations
    set_default_matplotlib_params(shape='rect_wide', l_side = 30)
    fig, ax = plt.subplots(2, 1)
    
    # Plot averages pm SEM of the top-p and worst-k natural images and of early 
    # and late generation windows.
    ax[0].barh(
        top_labels, 
        [val[0] for val in top_values], 
        xerr=[val[1] for val in top_values], 
        label='Top 3', 
        color='green'
    )
    ax[0].barh(
        bot_labels, 
        [val[0] for val in bot_values], 
        xerr=[val[1] for val in bot_values], 
        label='Bottom 3', 
        color='red'
    )
    ax[0].barh(
        list(gen_dict.keys()),
        [v[0] for v in gen_dict.values()],
        xerr=[v[1] for v in gen_dict.values()],
        label='Synthetic',
        color='black'
    )

    # Labels
    ax[0].set_xlabel('Average activation')
    ax[0].legend()
    
    
    # Replicate the same reasoning sorting by the best score
    sorted_lblA_bymax = sorted(lbl_acts.items(), key=lambda x: x[1][2]) # We sort for the 2 index, i.e. the max
    
    # Best and worst categories
    top_categories = sorted_lblA_bymax[-k:]
    bot_categories = sorted_lblA_bymax[:k]
    
    # Unpack labels and values
    top_labels, top_values = zip(*top_categories)
    bot_labels, bot_values = zip(*bot_categories)
    
    # Plot
    ax[1].barh(top_labels, [val[2] for val in top_values], label='Top 3',    color='green')
    ax[1].barh(bot_labels, [val[2] for val in bot_values], label='Bottom 3', color='red')

    # Labels
    ax[1].set_xlabel('Average activation')
    ax[1].legend()

    subplot_same_lims(ax, sel_axs = 'x')
    #customize_axes_bounds(ax[0])
    #customize_axes_bounds(ax[1])
    
    # Save
    if out_dir:
        out_fp = path.join(out_dir, 'scores_labels.png')
        logger.info(f'Saving score labels plot to {out_fp}')
        fig.savefig(out_fp, bbox_inches="tight")


def plot_optimizing_units(
    data: Dict[str, Dict[str, List[float]]],
    out_dir: str,
    logger: Logger = SilentLogger(),
    **kwargs
):
    # --- MACROS ---

    FIG_SIZE    = kwargs.get('FIG_SIZE', (20, 10))
    TITLE       = kwargs.get('TITLE', '')
    X_LABEL     = kwargs.get('X_LABEL', 'Random optimized units')
    Y_LABEL     = kwargs.get('Y_LABEL', 'Fitness')
    SAVE_FORMATS = kwargs.get('SAVE_FORMATS', ['svg'])

    PALETTE     = kwargs.get('PALETTE', 'husl')
    PLOT_ARGS   = kwargs.get('PLOT_ARGS', {'linestyle': '-', 'linewidth': 3, 'alpha': 0.5})
    ERRORBAR_ARGS = {'elinewidth': 2, 'capsize': 5, 'capthick': 2, 'alpha': 0.8}
    GRID_ARGS   = kwargs.get('GRID_ARGS', {'linestyle': '--', 'alpha': 0.7})

    # Font customization
    FONT_ARGS    = kwargs.get('FONT_ARGS', {'family': 'serif'})
    TICK_ARGS    = kwargs.get('TICK_ARGS', {'labelsize': 14, 'direction': 'out', 'length': 6, 'width': 2})
    LABEL_ARGS   = kwargs.get('LABEL_ARGS', {'fontsize': 16, 'labelpad': 10})
    TITLE_ARGS   = kwargs.get('TITLE_ARGS', {'fontsize': 20, 'fontweight': 'bold'})
    LEGEND_ARGS  = kwargs.get('LEGEND_ARGS', {
        'frameon': True, 'fancybox': True, 
        'framealpha': 0.7, 'loc': 'best', 'prop': {'family': FONT_ARGS['family'], 'size': 14}
    })


    # Define custom color palette with as many colors as layers
    custom_palette = sns.color_palette(PALETTE, len(data))

    for log_scale in [True, False]:

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for idx, (layer_name, neuron_scores) in enumerate(data.items()):

            # Compute mean and standard deviation for each x-value for both lines
            xs      = list(neuron_scores.keys())
            y_means = [np.mean(y)    for y in neuron_scores.values()]
            y_sem   = [    SEM(y)[0] for y in neuron_scores.values()]

            # Use custom color from the palette
            color = custom_palette[idx]

            # Plot line
            ax.plot(xs, y_means, label=layer_name, color=color, **PLOT_ARGS)

            # Error bars
            ax.errorbar(xs, y_means, yerr=y_sem, color=color, **ERRORBAR_ARGS, fmt='none')

        # Set labels, title, and legend
        ax.set_xlabel(X_LABEL, **LABEL_ARGS, **FONT_ARGS)
        ax.set_ylabel(Y_LABEL, **LABEL_ARGS, **FONT_ARGS)
        ax.set_title(f'{TITLE}', **TITLE_ARGS, **FONT_ARGS)
        ax.legend(**LEGEND_ARGS)
        
        if log_scale:
            ax.set_xscale('log')
        
        # Add grid
        ax.grid(True, **GRID_ARGS)
        
        # Customize ticks
        ax.tick_params(**TICK_ARGS)

        # Save or display  
        for fmt in SAVE_FORMATS:
            out_fp = os.path.join(out_dir, f'neurons_optimization_{"logscale" if log_scale else "natscale"}.{fmt}')
            logger.info(f'Saving score histogram plot to {out_fp}')
            fig.savefig(out_fp, bbox_inches='tight')

        
def multiexp_lineplot(out_df: DataFrame, ax: Axes | None = None, 
                      out_dir: str | None = None,
                      gr_vars: str|list[str] = ['layers', 'neurons'], 
                      y_var: str = 'scores', 
                      metrics: str|list[str] = ['mean', 'sem'],
                      logger: Logger | None = None):
    
    """Plot the final trend of a specific metric 
    across different population size of different layers.

    :param out_df: Results from the multiexperiment run 
    :type out_df: DataFrame
    :param ax: axis if you want to do multiexp_lineplot in a subplot, defaults to None
    :type ax: Axes | None, optional
    :param gr_vars: grouping variables, defaults to ['layers', 'neurons']
    :type gr_vars: str | list[str], optional
    :param y_var: variable you want to plot, defaults to 'scores'
    :type y_var: str, optional
    :param metrics: metrics you are interested to plot, defaults to ['mean', 'sem']
    :type metrics: str | list[str], optional
    """
    # Default logger
    logger = default(logger, SilentLogger())
    
    set_default_matplotlib_params(shape='rect_wide')
    # Define custom color palette with as many colors as layers
    layers = out_df['layers'].unique()
    #get len(layers) equally spaced colors from your colormap of interest
    custom_palette = cm.get_cmap('jet')
    colors = custom_palette(np.linspace(0, 1, len(layers)))

    # Group by the variables in gr_vars (default :'layers' and 'neurons')
    grouped = out_df.groupby(gr_vars)
    # Calculate metrics of interest (default: mean and sem) for all groupings
    result = grouped.agg(metrics) # type: ignore
    #if an axis is not defined create a new one
    if not(ax):
        fig, ax = plt.subplots(1)
    ax = cast(Axes, ax)
    #for each layer, plot the lineplot of interest
    for i,l in enumerate(layers): 
        layer = result.loc[l]
        if 'mean' in metrics and 'sem' in metrics:     
            ax.errorbar(layer.index.get_level_values('neurons'), layer[(y_var, 'mean')],
                yerr=layer[(y_var, 'sem')], label=l, color = colors[i])

            #TODO: think to other metrics to plot
    
    ax.set_xlabel('Neurons')
    ax.set_xscale('log')
    ax.set_ylabel(y_var.capitalize())
    ax.legend()
    #customize_axes_bounds(ax)
    # Save or display  
    if out_dir:
        fn = f'multiexp_{y_var}.svg'
        out_fp = path.join(out_dir, fn)
        logger.info(f'Saving {fn} to {out_fp}')
        fig.savefig(out_fp, bbox_inches="tight")


def save_stimuli_samples(
    stimuli_scores: Dict[str, List[Tuple[Scores, Codes]]],
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger(),
    fontdict = {}
): 
    
    ROUND = 3
    FIGSIZE = (10, 10)
    
    for name, stimuli_score in stimuli_scores.items():
    
        images_scores: List[Tuple[float, Stimuli]] = [
            (score.tolist()[0], generator(code)[0])
            for score, code in stimuli_score
        ]
        
        images_scores.sort(key=lambda x: x[0], reverse=True)
        
        superstimuli = [
            (to_pil_image(img_ten), f'Fitness: {round(fitness, ROUND)}')
            for fitness, img_ten in images_scores
        ]
        
        
        NROW    = math.ceil(math.sqrt(len(superstimuli)))
        NCOL    = len(superstimuli) // NROW
        
        if NROW * NCOL != len(superstimuli): NCOL += 1
        
        FONTDICT = overwrite_dict(
            a = {
                'fontsize'   : 10, 
                'fontweight' : 'medium',    # 'light', 'normal', 'medium', 'bold', 'heavy'
                'family'     : 'sans-serif' # 'serif', 'sans-serif', 'monospace', Times New Roman', 'Arial', 'Courier New'
            },
            b=fontdict
        )
        
        MAIN_TITLE_FT = fontdict.get('main_title', 24)
        
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(NROW, NCOL, figsize=FIGSIZE)
        
        if len(superstimuli) > 1: axes = axes.flatten()
        else                    : axes = [axes]

        for i, (img, title) in enumerate(superstimuli):
            axes[i].imshow(img)      # Display the image
            axes[i].set_title(title, fontdict=FONTDICT) # Add the title
            axes[i].axis('off')      # Remove the axis
        
        i += 1
        while i < NROW * NCOL:
            axes[i].axis('off')  # Hide the axis
            axes[i].imshow(PIL.Image.new('RGB', (1, 1), color='white'))  # Optionally add a white image
            i += 1
        
        plt.tight_layout(pad=2)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
        out_fp = path.join(out_dir, f'superstimuli_samples_{name}.png')
        
        logger.info(f'Saving superstimuli samples to {out_fp}')
        
        fig.savefig(out_fp)


def save_best_stimulus_per_variant(
    neurons_variant_codes: Dict[str, Dict[DeePSiMVariant, Tuple[Codes, Scores]]],
    gen_weights: str,
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    variants = list(list(neurons_variant_codes.values())[0].keys())
    
    # Load imagenet labels
    inet, _ = load_imagenet(logger=logger)
    
    # Load each generator per variant
    generators = {
        variant: DeePSiMGenerator(gen_weights, variant=variant)
        for variant in variants
    }
    
    # Create count to see which variant is the best
    variants_counts = {variant: 0 for variant in variants}
    
    for neurons, variant_codes in neurons_variant_codes.items():
    
        # Convert codes to images
        images = {
            variant: (score.tolist()[0], generators[variant](code)[0])
            for variant, (code, score) in variant_codes.items()
        }
        
        best_variant, best_score = variants[0], 0
        
        neuron_n = int(re.search(r"(\d+)=\[(\d+)\]", neurons).groups()[1]) # type: ignore
        label    = inet[neuron_n].name

        fig, axs = plt.subplots(1, len(variant_codes), figsize=(18, 3*len(variant_codes)))
        
        # Add a common general title
        fig.suptitle(label, fontsize=18)
        
        # Iterate over the images and scores
        for i, (variant, (score, image)) in enumerate(images.items()):
            
            if score > best_score:
                best_variant, best_score = variant, score
            
            ax = axs[i]

            # Plot the image
            img = image.detach().cpu().numpy().transpose(1, 2, 0)
            ax.imshow(img)

            # Set the title as the score
            ax.set_title(f"{variant} - score: {round(score, 3)}", fontsize=16)

            # Remove the axis ticks and labels
            ax.axis('off')
        
        variants_counts[best_variant] += 1
            
        # Adjust the spacing between subplots
        plt.tight_layout()
        
        # Save 
        out_fp = path.join(out_dir, f'{neuron_n}_variant_stimuli.png')
        
        logger.info(f'Saving stimuli per variant plot to {out_fp}')
        fig.savefig(out_fp, bbox_inches="tight")
        
        plt.close()
        
    
    # Create a barplot using variants_counts dictionary
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    custom_palette = sns.color_palette("husl", len(variants_counts))
    
    ax.bar(variants_counts.keys(), variants_counts.values(), color=custom_palette)
    
    ax.set_title('Best variant count per neuron')
    
    out_fp = path.join(out_dir, 'best_variant_count.png')
    
    logger.info(f'Saving best variant count plot to {out_fp}')
    
    fig.savefig(out_fp, bbox_inches="tight")
