import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import numpy as np
import pandas as pd
from typing import Literal,Union, List, Tuple
from matplotlib.axes import Axes
import torch



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
    
def plot_optimization_profile(optim, lab_col={'Synthetic':'k', 'Natural': 'g'}):
    set_default_matplotlib_params(shape='rect_wide', side = 30)
    fix, ax = plt.subplots(1, 2)
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
    plt.show()
    
    fix, ax = plt.subplots(1)
    
    score_nat =np.stack(optim._score_nat)
    score_gen =np.stack(optim._score)

    data_min = min(score_nat.min(), score_gen.min())
    data_max = max(score_nat.max(), score_gen.max())

    # Set the number of bins
    num_bins = 25  # like ponce 2019

    # Create histograms for both datasets with the same range and number of bins
    hnat = plt.hist(score_nat.flatten(), bins=num_bins, range=(data_min, data_max), alpha=1, label=lab[1],
                    density=True,edgecolor=col[1], linewidth =3)
    hgen_edges = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max), label= lab[0],
                 density=True,edgecolor= col[0], linewidth =3)
    hgen = plt.hist(score_gen.flatten(), bins=num_bins, range=(data_min, data_max), 
                    density=True ,color = col[0],edgecolor= col[0])

    for patch in hnat[-1]:
        patch.set_facecolor('none')

    for patch in hgen_edges[-1]:
        patch.set_facecolor('none')
        
    n_gens = score_gen.shape[0]
    for bound, patch in zip(hgen[1][1:], hgen[-1]):
        is_less = np.mean(score_gen < bound, axis = 1)
        alpha_col = np.sum(is_less*range(n_gens))/np.sum(range(n_gens))
        patch.set_alpha(alpha_col)
    
    plt.legend()
    plt.xlabel('Target Activation')
    plt.ylabel('Prob. density')
        
    plt.show()
    
        
        