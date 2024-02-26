import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import numpy as np
import pandas as pd
from typing import Literal,Union, List, Tuple
from matplotlib.axes import Axes
import torch
from os import path

from zdream.utils import SEMf



#TODO: Lorenzo, commentare e mettere in ordine ste funzioni di plotting

def Get_appropriate_fontsz(labels: List[str], figure_width: Union[float, int] = 0) -> float:
    """
    Calculate the appropriate fontsize for xticks (TO BE UPDATED FOR ALL) dynamically based on the number of labels and figure width.

    Parameters:
        labels (List[str]): List of label strings.
        figure_width (Union[float, int]): Width of the figure in inches. If 0, uses the default figure width.

    Returns:
        float: The calculated fontsize for xticks.
    """
    nr_labels = len(labels)
    max_label_length = max(len(label) for label in labels)
    if figure_width == 0:
        figure_width = plt.rcParams['figure.figsize'][0]
    Fontsz = ((figure_width / nr_labels) / max_label_length) / 0.0085 #0.0085 found empirically
    print('Best fontsize avoiding overlapping: '+str(Fontsz))
    return Fontsz


def set_default_matplotlib_params(side: float = 15, shape: Literal['square', 'rect_tall', 'rect_wide'] = 'square', sns_params=False, xlabels = []) -> None:
    """
    Set default Matplotlib parameters for better visualizations.

    Parameters:
    - side: Length of one side of the figure (default is 15).
    - shape: Shape of the figure ('square', 'rect_tall', or 'rect_wide', default is 'square').

    Returns:
    - graphic parameters (params + sns_params_dict if sns_params=True)
    """
    if shape == 'square':
        other_side = side
    elif shape == 'rect_wide':
        other_side = int(side * (2/3))
    elif shape == 'rect_tall':
        other_side = int(side * (3/2))
    else:
        raise ValueError("Invalid shape. Use 'square', 'rect_tall', or 'rect_wide'.")
    writing_sz = 50; standard_lw = 4; marker_sz = 20
    box_lw = 3; box_c = 'black'; median_lw = 4; median_c = 'red'
    subplot_distance = 0.3
    if not(xlabels==[]):
        writing_sz =  min(Get_appropriate_fontsz(xlabels, figure_width= side),writing_sz)
    params = {
        'figure.figsize': (side, other_side),
        'font.size': writing_sz,
        'axes.labelsize': writing_sz,
        'axes.titlesize': writing_sz,
        'xtick.labelsize': writing_sz,
        'ytick.labelsize': writing_sz,
        'legend.fontsize': writing_sz,
        'axes.grid': False,
        'grid.alpha': 0.2,
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
        'figure.subplot.hspace': subplot_distance, 
        'figure.subplot.wspace': subplot_distance 
    }
    plt.rcParams.update(params)
    if sns_params:
        sns_params_dict = {}
        sns_params_dict['medianprops']={"color":median_c,"linewidth":median_lw}; 
        sns_params_dict['whiskerprops']={"color":box_c,"linewidth":box_lw}; 
        sns_params_dict['boxprops']={"edgecolor":box_c,"linewidth":box_lw}
        return params,sns_params_dict

    return params
    
def Zoccolan_style_axes(ax):
    tick_positions_y = ax.get_yticklabels()
    tick_positions_x = ax.get_xticklabels()
    # Il primo tick sarà il primo elemento di tick_positions
    t1y = float(tick_positions_y[1].get_text().replace('−', '-'))
    t2y = float(tick_positions_y[-2].get_text().replace('−', '-'))

    t1x = float(tick_positions_x[1].get_text().replace('−', '-'))
    t2x = float(tick_positions_x[-2].get_text().replace('−', '-'))

    ax.spines['left'].set_bounds(t1y, t2y)
    ax.spines['bottom'].set_bounds(t1x, t2x)
    
def plot_optimization_profile(optim, lab_col={'Synthetic':'k', 'Natural': 'g'}, save_dir = None):
    set_default_matplotlib_params(shape='rect_wide', side = 30)
    fig_lp, ax = plt.subplots(1, 2)
    #in dubbio se usare questo schema
    lab = []
    col = []
    for  l,c in lab_col.items():
        lab.append(l)
        col.append(c)
        
    profile = ['best_shist', 'mean_shist']
    for i,k in enumerate(profile):
        ax[i].plot(optim.stats[k], label=lab[0], color=col[0])
        ax[i].plot(optim.stats_nat[k], label=lab[1], color = col[1])
        if k =='mean_shist':
            ax[i].fill_between(range(len(optim.stats[k])),optim.stats[k] - optim.stats['sem_shist'],
                            optim.stats[k] + optim.stats['sem_shist'], color=col[0], alpha=0.5) 
            ax[i].fill_between(range(len(optim.stats_nat[k])),optim.stats_nat[k] - optim.stats_nat['sem_shist'],
                optim.stats_nat[k] + optim.stats_nat['sem_shist'], color=col[1], alpha=0.5) 

        ax[i].set_xlabel('Generation cycles')
        ax[i].set_ylabel('Target Activation')
        ax[i].set_title(k.split('_')[0])
        ax[i].legend()
        Zoccolan_style_axes(ax[i])
    if save_dir:
        fig_lp.savefig(path.join(save_dir, f'scores_lineplot.png'), bbox_inches="tight")
    plt.show()

    
    fig_hist, ax = plt.subplots(1) #per ora plot di probability density lo faccio in un plot separato
    
    score_nat =np.stack(optim._score_nat)
    score_gen =np.stack(optim._score)

    data_min = min(score_nat.min(), score_gen.min())
    data_max = max(score_nat.max(), score_gen.max())

    num_bins = 25  # same nr of bins as in Ponce 2019

    # Create histograms for both datasets with the same range and number of bins
    hnat = plt.hist(score_nat.flatten(), bins=num_bins, range=(data_min, data_max), alpha=1, label=lab[1],
                    density=True,edgecolor=col[1], linewidth =3)
    #for generated images i want a alpha that is a function of the generation step. Given that alpha influences
    #also edge transparency, i had to separate the histogram in 2: one for edges, one for filling
    hgen_edg = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max), label= lab[0],
                 density=True,edgecolor= col[0], linewidth =3)
    hgen = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max), 
                    density=True ,color = col[0],edgecolor= col[0])

    for patch_nat, patch_hgen_edg in zip(hnat[-1], hgen_edg[-1]):
        patch_nat.set_facecolor('none')
        patch_hgen_edg.set_facecolor('none')
        
    n_gens = score_gen.shape[0]
    for up_bound, patch in zip(hgen[1][1:], hgen[-1]):
        is_less = np.mean(score_gen < up_bound, axis = 1)
        alpha_col = np.sum(is_less*range(n_gens))/np.sum(range(n_gens))
        patch.set_alpha(alpha_col)
    
    plt.legend()
    plt.xlabel('Target Activation')
    plt.ylabel('Prob. density')
    if save_dir:
        fig_hist.savefig(path.join(save_dir, f'scores_hist.png'), bbox_inches="tight")
        
    plt.show()
    
def plot_scores_by_cat(optim, lbls_presented, topk =3, save_dir = None):
    nat_scores = np.stack(optim._score_nat).flatten()
    gen_scores = np.stack(optim._score)
    nat_lbls = np.array(lbls_presented)
    unique_lbls, counts_lbls = np.unique(nat_lbls, return_counts=True)
    lbl_acts = {}
    for lb in unique_lbls:
        lb_scores = nat_scores[np.where(nat_lbls == lb)[0]]
        lbl_acts[lb] = (np.mean(lb_scores),SEMf(lb_scores), np.amax(lb_scores))
        
    sorted_lblA = sorted(lbl_acts.items(), key=lambda x: x[1][0])
    # Extracting top 3 and bottom 3 categories
    top_categories = sorted_lblA[-topk:]
    bottom_categories = sorted_lblA[:topk]
    # Unpacking top and bottom categories for plotting
    top_labels, top_values = zip(*top_categories)
    bottom_labels, bottom_values = zip(*bottom_categories)
    gen_dict = {'Early': (np.mean(gen_scores[:5,:]), SEMf(gen_scores[:5,:].flatten())),
                'Late': (np.mean(gen_scores[-5:,:]), SEMf(gen_scores[-5:,:].flatten()))}

    set_default_matplotlib_params(shape='rect_wide', side = 30)
    fig, ax = plt.subplots(2,1)
    ax[0].barh(top_labels, [val[0] for val in top_values], xerr=[val[1] for val in top_values], label='Top 3', color='green')
    ax[0].barh(bottom_labels, [val[0] for val in bottom_values], xerr=[val[1] for val in bottom_values], label='Bottom 3', color='red')
    ax[0].barh(list(gen_dict.keys()), [v[0] for _,v in gen_dict.items()], xerr=[v[1] for _,v in gen_dict.items()], label='Gens', color='black')

    ax[0].set_xlabel('Average Activation')
    ax[0].legend()
    
    sorted_lblA_bymax = sorted(lbl_acts.items(), key=lambda x: x[1][2])
    top_categories = sorted_lblA_bymax[-topk:]
    bottom_categories = sorted_lblA_bymax[:topk]
    top_labels, top_values = zip(*top_categories)
    bottom_labels, bottom_values = zip(*bottom_categories)
    ax[1].barh(top_labels, [val[2] for val in top_values], label='Top 3', color='green')
    ax[1].barh(bottom_labels, [val[2] for val in bottom_values], label='Bottom 3', color='red')
    
    if save_dir:
        fig.savefig(path.join(save_dir, f'scores_by_label.png'), bbox_inches="tight")
    plt.show()
        