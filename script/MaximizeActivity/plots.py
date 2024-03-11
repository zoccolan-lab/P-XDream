from os import path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from matplotlib.axes import Axes
from pandas import DataFrame
import seaborn as sns

from matplotlib import cm, pyplot as plt
import numpy as np
from numpy.typing import NDArray
from zdream.logger import Logger, MutedLogger
from zdream.optimizer import Optimizer
from zdream.utils.dataset import MiniImageNet
from zdream.utils.misc import SEM, default
from zdream.utils.plotting import customize_axes_bounds, set_default_matplotlib_params, subplot_same_lims



def plot_scores(
    scores: Tuple[NDArray, NDArray],
    stats:  Tuple[Dict[str, Any], Dict[str, Any]],
    style: Dict[str, Dict[str, str]] = {
        'gen': {'lbl': 'Synthetic', 'col': 'k'},
        'nat': {'lbl':   'Natural', 'col': 'g'}
    },
    num_bins: int  = 25, 
    out_dir: str | None = None,
    display_plots: bool = False,
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
    :param display_plots: If to display plots, default to False.
    :type out_dir: bool
    :param logger: Logger to log information relative to plot saving paths.
                   Defaults to None indicating no logging.
    :type logger: Logger | None
    '''

    # Preprocessing input
    scores_gen, scores_nat = scores
    stats_gen,  stats_nat  = stats

    logger = default(logger, MutedLogger())

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
    for i, k in enumerate(['best_shist', 'mean_shist']):
        
        # Lineplot of both synthetic and natural scores
        ax[i].plot(stats_gen[k], label=lbl_gen, color=col_gen)
        if use_nat:
            ax[i].plot(stats_nat[k], label=lbl_nat, color=col_nat)
        
        # When plotting mean values, add SEM shading
        if k =='mean_shist':
            for stat, col in zip(
                [stats_gen, stats_nat] if use_nat else [stats_gen], 
                [  col_gen,   col_nat] if use_nat else [  col_gen]
            ):
                ax[i].fill_between(
                    range(len(stat[k])),
                    stat[k] - stat['sem_shist'],
                    stat[k] + stat['sem_shist'], 
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
    else:
        plt.show()
    
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
    if display_plots:
        plt.show()


def plot_scores_by_cat(
    scores: Tuple[NDArray, NDArray], 
    lbls: List[int], 
    dataset: MiniImageNet,
    k: int = 3, 
    gens_window: int = 5,
    out_dir: str | None = None,
    display_plots: bool = False,
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
    :param display_plots: If to display plots, default to False.
    :type out_dir: bool
    :param logger: Logger to log information relative to plot saving paths.
                   Defaults to None indicating no logging.
    :type logger: Logger | None
    '''

    # Preprocessing input
    scores_gen, scores_nat = scores 

    logger = default(logger, MutedLogger())
    
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
    if display_plots:
        plt.show()


def plot_optimizing_units(
    multiexp_data: Dict[str, Any],
    out_dir: str | None = None,
    display_plots: bool = False,
    logger: Logger | None = None
):
    #OBSOLETE: multiexp_lineplot is a generalized version of plot_optimizing_units
    '''
    Plot the final score trend across different population size of optimizing neurons
    for different layers
    
    :param multiexp_data: Results from the multiexperiment run as a mapping to equal-length lists 
                          containing information relative to number of optimization neurons, scores,
                          layers and optimization steps.
    :param out_dir: Directory where to save output plots, default is None indicating not to save.
    :type out_dir: str | None
    :param display_plots: If to display plots, default to False.
    :type out_dir: bool
    :param logger: Logger to log information relative to plot saving paths.
                   Defaults to None indicating no logging.
    :type logger: Logger | None
    '''
    
    # Auxiliar functions
    def find_indices(lst, value): return [i for i, item in enumerate(lst) if item == value]

    def extract_layer(el):
        ''' Auxiliar function to extract layer names from a list element'''
        if len(el) > 1:
            raise ValueError(f'Expected to record from a single layer, multiple where found: {el}')
        return el[0]

    # Default logger
    logger = default(logger, MutedLogger())

    # Check same gen
    all_gen = set(multiexp_data['num_gens'])
    if len(all_gen) > 1:
        err_msg = f'Experiments were requires to have the same number of iterations, but multiple found {all_gen}'
        raise ValueError(err_msg)
    gen = list(all_gen)[0]
    
    # Cast data to raw type lists
    data = {
        'scores'  : list(np.concatenate(multiexp_data['score'])),
        'neurons' : multiexp_data['neurons'],
        'layers'  : [extract_layer(el) for el in multiexp_data['layer']]
    }

    # Organize data in a mapping {layer: {neurons: [score1, score2, ...]}}
    unique_layers = set(data['layers'])
    combined_data = defaultdict(dict)
    for layer in unique_layers:
        ii = find_indices(data['layers'], layer)
        neurons = [data['neurons'][i] for i in ii]
        scores_  = [data['scores' ][i] for i in ii]
        unique_neurons = sorted(set(neurons))
        for neuron in unique_neurons:
            jj = find_indices(neurons, neuron)
            scores = [scores_[j] for j in jj]
            combined_data[layer][neuron] = np.array(scores)
            
    

    # Plot
    fig, ax = plt.subplots(1)
    set_default_matplotlib_params(shape='rect_wide')

    # Define custom color palette with as many colors as layers
    custom_palette = sns.color_palette("husl", len(combined_data))
    
    

    for idx, (label, ys) in enumerate(combined_data.items()):

        # Compute mean and standard deviation for each x-value for both lines
        xs = list(ys.keys())
        y_means = [np.mean(y) for y in ys.values()]
        y_sem = [SEM(y) for y in ys.values()]

        # Use custom color from the palette
        color = custom_palette[idx]

        # Plot line
        ax.plot(xs, y_means, label=label, color=color)

        # Plot error bars for standard deviation for both lines
        ax.errorbar(xs, y_means, yerr=y_sem, color=color)

    # Set labels, title and legend
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Label')
    ax.set_title(f'Final score varying optimization neurons in {gen} epochs. ')
    ax.legend()

    # Set x-axis ticks to integer values and only where the points are
    ax.set_xticks(xs)
    #customize_axes_bounds(ax)

    # Save or display  
    if out_dir:
        out_fp = path.join(out_dir, 'neurons_optimization.png')
        logger.info(f'Saving score histogram plot to {out_fp}')
        fig.savefig(out_fp, bbox_inches="tight")
    if display_plots:
        plt.show()
        
        
def multiexp_lineplot(out_df: DataFrame, ax: Axes | None = None, 
                      gr_vars: str|list[str] = ['layers', 'neurons'], 
                      y_var: str = 'scores', 
                      metrics: str|list[str] = ['mean', 'sem']):
    
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
    
    set_default_matplotlib_params(shape='rect_wide')
    # Define custom color palette with as many colors as layers
    layers = out_df['layers'].unique()
    #get len(layers) equally spaced colors from your colormap of interest
    custom_palette = cm.get_cmap('jet')
    colors = custom_palette(np.linspace(0, 1, len(layers)))

    # Group by the variables in gr_vars (default :'layers' and 'neurons')
    grouped = out_df.groupby(gr_vars)
    # Calculate metrics of interest (default: mean and sem) for all groupings
    result = grouped.agg(metrics)
    #if an axis is not defined create a new one
    if not(ax):
        fig, ax = plt.subplots(1)
    #for each layer, plot the lineplot of interest
    for i,l in enumerate(layers): 
        layer = result.loc[l]
        if 'mean' in metrics and 'sem' in metrics:     
            ax.errorbar(layer.index.get_level_values('neurons'), layer[(y_var, 'mean')],
                yerr=layer[(y_var, 'sem')], label=l, color = colors[i])
            #TODO: think to other metrics to plot
            
    ax.set_xlabel('Neurons')
    ax.set_ylabel(y_var.capitalize())
    ax.legend()
    customize_axes_bounds(ax)
    
    # Show plot
    plt.show()