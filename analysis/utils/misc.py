import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from typing import Dict, List
from zdream.utils.logger import Logger, SilentLogger

# --- LOGGING ---

def start(logger: Logger, name: str):
    
    logger.info(mess=name)
    logger.formatting = lambda x: f'> {x}'

def end(logger: Logger):
    
    logger.info(mess='Done')
    logger.reset_formatting()
    logger.info(mess='')
    
# --- PLOTTING ---

def plot(
    data      : Dict[str, List[int]] | Dict[str, List[float]],
    ylabel    : str,
    title     : str,
    file_name : str,
    out_dir   : str,
    logger    : Logger = SilentLogger()
):
    '''
    Plot data in a boxplot and violinplot and save them in the output directory

    :param data: Data to plot indexed by cluster type.
    :type data: Dict[str, List[int]] | Dict[str, List[float]]
    :param ylabel: Label for the y axis.
    :type ylabel: str
    :param title: Title of the plot.
    :type title: str
    :param file_name: Name of the file to save the plot.
    :type file_name: str
    :param out_dir: Output directory.
    :type out_dir: str
    :param logger: Logger to log the process, defaults to SilentLogger().
    :type logger: Logger, optional
    '''
    
    SNS_PALETTE = 'husl'
    FIGSIZE     = (10, 6)
    
    palette = sns.set_palette(SNS_PALETTE)
    
    for plot_ty in ['boxplot', 'violinplot']:
    
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        data_values = list(data.values())
        
        if plot_ty == 'boxplot': sns.boxplot   (data=data_values, ax=ax, palette=palette)
        else:                    sns.violinplot(data=data_values, ax=ax, palette=palette)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(data_values)))
        ax.set_xticklabels(data.keys())
        ax.tick_params(axis='x', rotation=45) 
        plt.tight_layout() 
        
        out_fp = os.path.join(out_dir, f'{file_name}_{plot_ty}.svg')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)