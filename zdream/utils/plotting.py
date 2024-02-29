#TODO: Lorenzo, commentare e mettere in ordine ste funzioni di plotting

from os import path
import re
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from typing import List, Literal, Union, Any
import numpy as np
from zdream.utils.misc import SEMf
from zdream.optimizer import Optimizer


def Get_appropriate_fontsz(labels: List[str], figure_width: Union[float, int] = 0) -> float:
    """
    Calculate the appropriate fontsize for xticks dynamically based on the number of labels and figure width.
    
    :param labels: List of label strings
    :type labels: List[str]
    :param figure_width: Width of the figure in inches. If 0, uses the default figure width.
    :type figure_width: Union[float, int]
    
    :return: The calculated fontsize for xticks.
    :rtype: float
    """
    
    #compute the length of the longest label present
    max_label_length = max(len(label) for label in labels)
    #in case where figure_width is not specified, use default figsize
    if figure_width == 0:
        figure_width = plt.rcParams['figure.figsize'][0]
    #compute the appropriate fontsize to avoid overlaps (NOTE: 0.0085 was found empirically)
    Fontsz = ((figure_width / len(labels)) / max_label_length) / 0.0085 
    print('Best fontsize avoiding overlapping: '+str(Fontsz))
    return Fontsz


def set_default_matplotlib_params(l_side: float = 15, shape: Literal['square', 'rect_tall', 'rect_wide'] = 'square', 
                                  xlabels:list[str] = []
                                  ) -> dict[str, Any]:
    """
    Set your default Matplotlib parameters.
    
    :param l_side: Length of the lower side of the figure (default is 15).
    :type l_side: float
    :param shape: Shape of the figure ('square', 'rect_tall', or 'rect_wide', default is 'square').
    :type shape: Literal['square', 'rect_tall', 'rect_wide'] 
    :param xlabels: List of xlabels to apply Get_appropriate_fontsz
    :type xlabels: list[str]
    
    :return: default graphic parameters
    :rtype: dict[str, Any]
    """
    #set the dimension of the other side of the figure based
    #on the shape
    if shape == 'square':
        other_side = l_side
    elif shape == 'rect_wide':
        other_side = l_side * (2/3)
    elif shape == 'rect_tall':
        other_side = l_side * (3/2)
    else:
        raise ValueError("Invalid shape. Use 'square', 'rect_tall', or 'rect_wide'.")
    
    #default params
    writing_sz = 35; standard_lw = 4; marker_sz = 20
    box_lw = 3; box_c = 'black'; median_lw = 4; median_c = 'red'
    subplot_distance = 0.3; axes_lw = 3; tick_length = 6
    
    if not(xlabels==[]):
        writing_sz =  min(Get_appropriate_fontsz(xlabels, figure_width= l_side),writing_sz)
    params = {
        'figure.figsize': (l_side, other_side),
        'font.size': writing_sz,
        'axes.labelsize': writing_sz,
        'axes.titlesize': writing_sz,
        'xtick.labelsize': writing_sz,
        'ytick.labelsize': writing_sz,
        'legend.fontsize': writing_sz,
        'axes.grid': False,
        'grid.alpha': 0.4,
        'lines.linewidth': standard_lw,
        'lines.linestyle': '-',
        'lines.markersize': marker_sz,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'errorbar.capsize': standard_lw,
        'boxplot.boxprops.linewidth': box_lw,
        'boxplot.boxprops.color': box_c,
        'boxplot.whiskerprops.linewidth': box_lw,
        'boxplot.whiskerprops.color': box_c,
        'boxplot.medianprops.linewidth': median_lw,
        'boxplot.medianprops.color': median_c,
        'boxplot.capprops.linewidth': box_lw,
        'boxplot.capprops.color': box_c,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': axes_lw,
        'xtick.major.width': axes_lw,
        'ytick.major.width': axes_lw,
        'xtick.major.size': tick_length,
        'ytick.major.size': tick_length,
        'figure.subplot.hspace': subplot_distance,
        'figure.subplot.wspace': subplot_distance
    }
    plt.rcParams.update(params)

    return params


def Zoccolan_style_axes(ax: Axes):
    """
    Modify axes in the way that Davide likes.
    
    :param ax: axis you want to modify
    :type ax: Axes
    """
    #get tick labels
    tick_positions_y = ax.get_yticklabels()
    tick_positions_x = ax.get_xticklabels()
    #select the new upper and lower bounds for both axes
    t1y = tick_positions_y[1].get_text()
    t2y = tick_positions_y[-2].get_text()
    t1x = tick_positions_x[1].get_text()
    t2x = tick_positions_x[-2].get_text()
    
    #set the new axes bound according to the type of labels (i.e. numeric or categorical)
    for side, (low_b,up_b) in zip(['left','bottom'], ((t1y, t2y), (t1x, t2x))):
        if re.sub(r'[−.]', '',low_b).isdigit():
            #case 1(numeric): just set as bounds the 1st and last thicks as floats
            ax.spines[side].set_bounds(float(low_b.replace('−','-')),float(up_b.replace('−','-')))
        else:
            #case 2(categorical): set as bounds the index of first and last thicks
            ticks = tick_positions_y if side=='left' else tick_positions_x
            low_b = 0; up_b = len(ticks)-1
            ax.spines[side].set_bounds(low_b,up_b)


def plot_optimization_profile(optim: Optimizer, lab_col : dict[str, dict[str, str| list[float]]]
                              ={'gen':{'lbl':'Synthetic','col':'k'}, 'nat':{'lbl':'Natural','col':'g'}},
                              num_bins = 25, save_dir: str |None = None):
    """
    Plot the progression of the optimization. The first plot displays the activation of the best image per 
    optimization step (left) and the average activation pm SEM per iteration (right), for both natural and synthetic
    images. The second plot displays the histograms of activations for both natural and synthetic images.
    
    :param optim: Optimizer. We will use in particular its attributes "scores"
    :type optim: Optimizer
    :param lab_col: dictionary reporting the labels and colors of natural and generated images 
                    (by default, nats are green and gens are black)
    :type lab_col: dict[str, dict[str, str| list[float]]]
    :param save_dir: Directory where to save the plots (default is None)
    :type save_dir: str |None
    """
    
    def_params = set_default_matplotlib_params(shape='rect_wide', l_side = 30)
    alpha = def_params['grid.alpha'] #alpha must be explicited to fill_between
    
    #PLOT 1) LINEPLOT: BEST AND AVERAGE ACTIVATIONS
    fig_lp, ax = plt.subplots(1, 2)
    
    #define label and colors for both generated and natural images
    lbl_gen = lab_col['gen']['lbl']; col_gen = lab_col['gen']['col']
    lbl_nat = lab_col['nat']['lbl']; col_nat = lab_col['nat']['col']

    profile = ['best_shist', 'mean_shist'] #i.e. the two types of subplot
    for i,k in enumerate(profile):
        #lineplots of best or avg
        ax[i].plot(optim.stats[k], label=lbl_gen, color=col_gen)
        ax[i].plot(optim.stats_nat[k], label=lbl_nat, color = col_nat)
        #when plotting the means, do SEM shading
        if k =='mean_shist':
            ax[i].fill_between(range(len(optim.stats[k])),optim.stats[k] - optim.stats['sem_shist'],
                            optim.stats[k] + optim.stats['sem_shist'], color=col_gen, alpha=alpha)
            ax[i].fill_between(range(len(optim.stats_nat[k])),optim.stats_nat[k] - optim.stats_nat['sem_shist'],
                optim.stats_nat[k] + optim.stats_nat['sem_shist'], color=col_nat, alpha=alpha)
        #plot details
        ax[i].set_xlabel('Generation cycles')
        ax[i].set_ylabel('Target Activation')
        ax[i].set_title(k.split('_')[0].capitalize())
        ax[i].legend()
        Zoccolan_style_axes(ax[i])
    #if a saving dir is passed, then save the plot    
    if save_dir:
        fig_lp.savefig(path.join(save_dir, f'scores_lineplot.png'), bbox_inches="tight")
    plt.show()
    

    #PLOT 2) SCORES HISTOGRAM/DISTRIBUTION
    fig_hist, ax = plt.subplots(1) 
    #get all scores of all images
    score_nat =np.stack(optim._score_nat)
    score_gen =np.stack(optim._score)
    
    data_min = min(score_nat.min(), score_gen.min())
    data_max = max(score_nat.max(), score_gen.max())

    num_bins = 25  # same nr of bins as in Ponce 2019

    # Create histograms for both datasets with the same range and number of bins
    hnat = plt.hist(score_nat.flatten(), bins=num_bins, range=(data_min, data_max), alpha=1, label=lbl_nat,
                    density=True,edgecolor=col_nat, linewidth =3)
    #for generated images i want a alpha that is a function of the generation step. Given that alpha influences
    #also edge transparency, i had to separate the histogram in 2: one for edges, one for filling
    hgen_edg = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max), label= lbl_gen,
                 density=True,edgecolor= col_gen, linewidth =3)
    hgen = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max),
                    density=True ,color = col_gen,edgecolor= col_gen)
    #for both the natural imgs histogram and for the one of gen edges, i set the columns to be transparent
    for patch_nat, patch_hgen_edg in zip(hnat[-1], hgen_edg[-1]):
        patch_nat.set_facecolor('none')
        patch_hgen_edg.set_facecolor('none')
    
    n_gens = score_gen.shape[0]
    #here i set the alpha values for each column of the hgen histogram.
    #i iterate over each column, taking its upper bound and the related graphic object
    for up_bound, patch in zip(hgen[1][1:], hgen[-1]):
        #i compute the probability for each generation of being less than the upper bound
        #of the column of interest
        is_less = np.mean(score_gen < up_bound, axis = 1)
        #the alpha is the weighted average of these probabilities, where the weights are
        #the indexes of the generations. The later the generation, the higher its weight 
        # (the stronger the shade of black).
        #POSSIBLE ISSUE: is weighting too steep in this way?
        alpha_col = np.sum(is_less*range(n_gens))/np.sum(range(n_gens))
        patch.set_alpha(alpha_col)
    #plot details
    plt.legend()
    plt.xlabel('Target Activation')
    plt.ylabel('Prob. density')
    Zoccolan_style_axes(ax)
    if save_dir:
        fig_hist.savefig(path.join(save_dir, f'scores_hist.png'), bbox_inches="tight")

    plt.show()


def plot_scores_by_cat(optim: Optimizer, lbls_presented: list[bool], topk: int =3, 
                       n_gens_considered: int = 5, save_dir: str|None = None):
    """
    Plot illustrating the average pm SEM scores of the topk best and worst natural images categories.
    Moreover, it shows the average pm SEM scores of generated images in the first and last n_gens_considered.
    
    :param optim: Optimizer. We will use in particular its attributes "scores"
    :type optim: Optimizer
    :param lbls_presented: List of the labels of the natural images presented during the
                           optimization.
    :type lbls_presented: list[bool]
    :param topk: number of top and bottom nat imgs classes by score considered. Default is 3
    :type topk: int
    :param n_gens_considered: number of generations considered at the beginning and end considered. Default is 5
    :type n_gens_considered: int
    :param save_dir: Directory where to save the plots (default is None)
    :type save_dir: str |None
    """
    #organize scores and labels as np arrays. nat_scores are flattened to be
    #easily indexed by the nat_lbls vector
    nat_scores = np.stack(optim._score_nat).flatten()
    gen_scores = np.stack(optim._score)
    nat_lbls = np.array(lbls_presented)
    #get the unique labels present in the dataset
    unique_lbls, _ = np.unique(nat_lbls, return_counts=True)
    #gather in the lbl_acts dictionary the score statistics of interest
    # (mean, SEM and max) for each of the labels
    lbl_acts = {}
    for lb in unique_lbls:
        lb_scores = nat_scores[np.where(nat_lbls == lb)[0]]
        lbl_acts[lb] = (np.mean(lb_scores),SEMf(lb_scores), np.amax(lb_scores))
    #sort the scores by the average (we take the vals of the dict (x[1]) and 
    # we select the first element of each tuple ([0]))
    sorted_lblA = sorted(lbl_acts.items(), key=lambda x: x[1][0])
    # Extracting top 3 and bottom 3 categories
    top_categories = sorted_lblA[-topk:]
    bottom_categories = sorted_lblA[:topk]
    # Unpacking top and bottom categories for plotting
    top_labels, top_values = zip(*top_categories)
    bottom_labels, bottom_values = zip(*bottom_categories)
    #in gen_dict i define the mean and std of early and late generation epochs
    gen_dict = {'Early': (np.mean(gen_scores[:n_gens_considered,:]), SEMf(gen_scores[:n_gens_considered,:].flatten())),
                'Late': (np.mean(gen_scores[-n_gens_considered:,:]), SEMf(gen_scores[-n_gens_considered:,:].flatten()))}
    
    #Now i plot the data
    set_default_matplotlib_params(shape='rect_wide', l_side = 30)
    fig, ax = plt.subplots(2,1)
    #first i plot the averages pm sem of the best and worst topk nat images and of early and late gens
    ax[0].barh(top_labels, [val[0] for val in top_values], xerr=[val[1] for val in top_values], label='Top 3', color='green')
    ax[0].barh(bottom_labels, [val[0] for val in bottom_values], xerr=[val[1] for val in bottom_values], label='Bottom 3', color='red')
    ax[0].barh(list(gen_dict.keys()), [v[0] for _,v in gen_dict.items()], xerr=[v[1] for _,v in gen_dict.items()], label='Gens', color='black')

    ax[0].set_xlabel('Average activation')
    ax[0].legend()
    Zoccolan_style_axes(ax[0])
    
    #now i plot the best scores, with a logic identical to what i did for average
    sorted_lblA_bymax = sorted(lbl_acts.items(), key=lambda x: x[1][2])
    top_categories = sorted_lblA_bymax[-topk:]
    bottom_categories = sorted_lblA_bymax[:topk]
    top_labels, top_values = zip(*top_categories)
    bottom_labels, bottom_values = zip(*bottom_categories)
    ax[1].barh(top_labels, [val[2] for val in top_values], label='Top 3', color='green')
    ax[1].barh(bottom_labels, [val[2] for val in bottom_values], label='Bottom 3', color='red')
    Zoccolan_style_axes(ax[1])
    if save_dir:
        fig.savefig(path.join(save_dir, f'scores_by_label.png'), bbox_inches="tight")
    plt.show()