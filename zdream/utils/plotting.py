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

