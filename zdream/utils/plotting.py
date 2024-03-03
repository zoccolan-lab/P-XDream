#TODO: Lorenzo, commentare e mettere in ordine ste funzioni di plotting

from os import path
import re
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from typing import List, Literal, Dict, Any
import numpy as np
from zdream.utils.dataset import MiniImageNet
from zdream.utils.misc import SEM, default
from zdream.optimizer import Optimizer

# --- DEFAULT PLOTTING PARAMETERS ----

_Shapes = Literal['square', 'rect_tall', 'rect_wide']

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
        
        # Case 2: Categorical - Set first and last thicks bounds as index
        else:
            ticks = tick_positions_y if side == 'left' else tick_positions_x
            low_b = 0
            up_b  = len(ticks)-1
        
        ax.spines[side].set_bounds(low_b, up_b)

# --- PLOTS ---

def plot_scores(
    optim: Optimizer,
    lab_col : Dict[str, Dict[str, str]] = {
        'gen': {'lbl': 'Synthetic', 'col': 'k'},
        'nat': {'lbl':   'Natural', 'col': 'g'}
    },
    num_bins: int  = 25, 
    out_dir: str | None = None
):
    '''
    Plot two views of scores trend through optimization steps.
    
    1. The first plot displays the score of the best stimulus per optimization step (left) 
       and the average stimuli score pm SEM per iteration (right), for both natural and synthetic images. 

    2. The second plot displays the histograms of scores for both natural and synthetic images.
    
    :param optim: Optimizer collecting stimuli scores across iterations for both synthetic and natural images.
    :type optim: Optimizer
    :param lab_col: Style dictionary setting labels and colors of synthetic and natural images.
                    They default to black for synthetic for green for natural.
    :type lab_col: Dict[str, Dict[str, str]]
    :param num_bins: Number of bins to display in the histogram, defaults to 25.
    :type num_bins: int
    :param out_dir: Directory where to save output plots, default is None indicating not to save
                     but to display with function call.
    :type out_dir: str | None
    '''
    
    # Retrieve default parameters and retrieve `alpha` parameter
    def_params = set_default_matplotlib_params(shape='rect_wide', l_side = 30)
    
    # PLOT 1. BEST and AVG SCORE TREND
    fig_trend, ax = plt.subplots(1, 2)
    
    # Define label and colors for both generated and natural images
    lbl_gen = lab_col['gen']['lbl']; col_gen = lab_col['gen']['col']
    lbl_nat = lab_col['nat']['lbl']; col_nat = lab_col['nat']['col']

    # We replicate the same reasoning for the two subplots by accessing 
    # the different key of the score dictionary whether referring to max or mean.
    for i, k in enumerate(['best_shist', 'mean_shist']):
        
        # Lineplot of both synthetic and natural scores
        ax[i].plot(optim.stats    [k], label=lbl_gen, color=col_gen)
        ax[i].plot(optim.stats_nat[k], label=lbl_nat, color=col_nat)
        
        # When plotting mean values, add SEM shading
        if k =='mean_shist':
            for stats, col in zip([optim.stats, optim.stats_nat], [col_gen, col_nat]):
                ax[i].fill_between(
                    range(len(stats[k])),
                    stats[k] - stats['sem_shist'],
                    stats[k] + stats['sem_shist'], 
                    color=col, alpha=def_params['grid.alpha']
                )
                
        # Names
        ax[i].set_xlabel('Generation cycles')
        ax[i].set_ylabel('Target scores')
        ax[i].set_title(k.split('_')[0].capitalize())
        ax[i].legend()
        customize_axes_bounds(ax[i])
    
    # Save or display  
    if out_dir:
        out_fp = path.join(out_dir, 'scores_trend.png')
        fig_trend.savefig(out_fp, bbox_inches="tight")
    else:
        plt.show()
    
    # PLOT 2. SCORES HISTOGRAM 
    fig_hist, ax = plt.subplots(1) 
    
    # Stack scores stimuli
    score_nat = np.stack(optim._scores_nat)
    score_gen = np.stack(optim._scores)
    
    # Compute min and max values
    data_min = min(score_nat.min(), score_gen.min())
    data_max = max(score_nat.max(), score_gen.max())

    # Create histograms for both synthetic and natural with the same range and bins
    hnat = plt.hist(
        score_nat.flatten(),
        bins=num_bins,
        range=(data_min, data_max),
        alpha=1, 
        label=lbl_nat,
        density=True,
        edgecolor=col_nat, 
        linewidth=3
    )
    
    hgen = plt.hist(
        score_gen.flatten(),
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
        score_gen.flatten(), 
        bins=num_bins, 
        range=(data_min, data_max), 
        label= lbl_gen,
        density=True,
        edgecolor=col_gen,
        linewidth=3
    )
    
    # Set transparent columns for natural images and edges of synthetic ones
    for bins in [hnat, hgen_edg]:
        for bin in bins[-1]: # type: ignore
            bin.set_facecolor('none')
    
    # Retrieve number of iterations
    n_gens = score_gen.shape[0]
    
    # Set alpha value of each column of the hgen histogram.
    # Iterate over each column, taking its upper bound and its histogram bin
    for up_bound, bin in zip(hgen[1][1:], hgen[-1]): # type: ignore
        
        # Compute the probability for each generation of being
        # less than the upper bound of the bin of interest
        is_less = np.mean(score_gen < up_bound, axis = 1)
        
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
    customize_axes_bounds(ax)
    
    # Save or display  
    if out_dir:
        out_fp = path.join(out_dir, 'scores_histo.png')
        fig_hist.savefig(out_fp, bbox_inches="tight")
    else:
        plt.show()


def plot_scores_by_cat(
    optim: Optimizer, 
    lbls: List[int], 
    dataset: MiniImageNet,
    k: int =3, 
    gens_window: int = 5,
    out_dir: str|None = None
):
    '''
    Plot illustrating the average pm SEM scores of the top-k and bottom-k categories of natural images.
    Moreover it also shows the average pm SEM scores of synthetic images within the first and last
    iteration of a considered generation window.
    
    :param optim: Optimizer collecting stimuli score across generations.
    :type optim: Optimizer
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
    :param out_dir: Directory where to save output plots, default is None indicating not to save
                     but to display with function call.
    :type out_dir: str | None
    '''
    
    # Cast scores and labels as arrays.
    # NOTE: `nat_scores are`` flattened to be more easily 
    #       indexed by the nat_lbls vector
    nat_scores = np.stack(optim._scores_nat).flatten()
    gen_scores = np.stack(optim._scores)
    
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
    customize_axes_bounds(ax[0])
    
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
    customize_axes_bounds(ax[1])
    
    # Save
    if out_dir:
        out_fp = path.join(out_dir, 'scores_labels.png')
        fig.savefig(out_fp, bbox_inches="tight")
    else:
        plt.show()