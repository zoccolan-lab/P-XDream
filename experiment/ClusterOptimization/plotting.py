from collections import defaultdict
import math
from os import path
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from torchvision.transforms.functional import to_pil_image

from zdream.generator import Generator
from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.message import ZdreamMessage
from zdream.utils.misc import SEM, concatenate_images
from zdream.utils.types import Codes


def plot_weighted(
    cluster_idx: List[int],
    weighted: List[bool],
    scores: List[List[NDArray]],
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    # Line and background
    COLORS = {
        'arithmetic' :  ('#229954', '#22995480'),
        'weighted'   :  ('#A569BD', '#A569BD80')
    }
    
    # Compute means and max of population for each generation
    means, maxs = (
        [
            [op(iter_score) for iter_score in exp_score] 
            for exp_score in scores
        ] for op in [np.mean, np.max]
    )
    
    # Organize data for computation
    # cluster_id: statistic (mean or max) : score type (weighted or arithmetic): scores
    clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, weight, scores in zip(cluster_idx, weighted, zip(means, maxs)): # type: ignore
        mean_, max_ = scores
        w = "weighted" if weight else 'arithmetic'
        clusters[idx]['mean'][w].append(mean_)
        clusters[idx]['max' ][w].append(max_)
    
    # Compute mean and standard deviation across samples
    clusters = {
        idx: {
            stat: {
                scr_type: list(
                    zip(*[(np.mean(iter_samples) ,SEM(iter_samples)) 
                    for iter_samples in list(zip(*generations))])
                )
                for scr_type, generations in scores.items()
            }
            for stat, scores in cluster_info.items()
        }
        for idx, cluster_info in clusters.items()
    }
    
    # Plot
    for clu_idx, cluster_info in clusters.items():
    
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        
        for i, (stat, generations) in enumerate(cluster_info.items()):
            
            for scr_type, scores in generations.items():
                
                means, stds = scores
                
                means = np.array(means)
                stds  = np.array(stds)
                
                col1, col2 = COLORS[scr_type]
                
                # Line
                axes[i].plot(range(len(means)), means, color=col1, label=scr_type)

                # IC
                axes[i].fill_between(
                    range(len(stds)),
                    means - stds,
                    means + stds, 
                    color = col2
                )
            
            axes[i].set_xlabel('Generations')
            axes[i].set_ylabel(f'{stat.capitalize()} activation')
            axes[i].set_title('')
            axes[i].grid(True)
            axes[i].legend()
        
        fig.suptitle(f'Cluster {clu_idx}')
        
        out_fp = path.join(out_dir, f'cluster_{clu_idx}.png')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)


def plot_scr(
    data: Dict[int, Dict[str, List[float]]],
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    # Color, offset
    STYLE = {
        'cluster'    : ('#DC7633',  .0), 
        'random'     : ('#3498DB', -.1), 
        'random_adj' : ('#52BE80', +.1)
    }
    EDGE_COLOR = 'black'
    
    # Lineplots points to be computed
    linesX = {k: [] for k in STYLE}
    linesY = {k: [] for k in STYLE}
    
    errorbars,   ax1 = plt.subplots(figsize=(16, 8))
    violinplots, ax2 = plt.subplots(figsize=(16, 8))
    
    # Plot
    for cluster_idx, cluster_scores in data.items():
        for scr_type, y in cluster_scores.items():
            
            color, offset = STYLE[scr_type]
            
            # ERROR BARS
            x = cluster_idx + offset * 0.2
            y_bar = np.mean(y)
            ax1.errorbar(x, y_bar, yerr=SEM(y), color=color, label=scr_type, ecolor=color, fmt='o', markersize=8, capsize=10)
            
            linesX[scr_type].append(x)
            linesY[scr_type].append(y_bar)
            
            # VIOLIN PLOTS
            x = cluster_idx + offset * 0.5

            violin_parts = ax2.violinplot(
                dataset=y,
                positions=[x],
                widths=0.1, 
                showmeans=False,
                showextrema=False
            )
            
            for pc in violin_parts['bodies']:  # type: ignore
                pc.set_color(color)
                pc.set_alpha(1)
                pc.set_edgecolor(EDGE_COLOR)

    # Customizing legend
    legend_labels = [
        plt.Line2D(  # type: ignore
            [0], [0], color=color, lw=4, label=scr_type)
        for scr_type, (color, _) in STYLE.items()]
    
    # Line plot for ERROR BARS
    for name, (color, _) in STYLE.items():
        ax1.plot(linesX[name], linesY[name], color=color, marker=',')
    
    # Customize axes
    for ax in [ax1, ax2]:    
        
        ax.legend(handles=legend_labels, title='Scr Types')
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Scores')
        ax.set_title('Scoring Types comparison')
        ax.set_xticks(list(data.keys()))
        
    # Save figures
    for fig, name in [
        (errorbars,   'errorbars'),
        (violinplots, 'violins'), ]:
        out_fp = path.join(out_dir, f'scr_types_{name}.png')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)


def plot_activations(
    activations: Dict[str, NDArray],
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    COLORS = ['#0013D6', '#E65c00','#36A900']
    
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))

    for i, (name, act) in enumerate(activations.items()):

        color = COLORS[i]

        # 1 plot
        for i in range(act.shape[1]):
            axes[0].plot(act[:, i], color=color, linewidth=0.8)

        axes[0].set_xlabel('Generations')
        axes[0].set_ylabel('Unit activation')
        axes[0].set_title('Individual neurons activations')
        axes[0].grid(True)

        # 2 plot
        means = np.mean(act, axis=1)
        stds  =     SEM(act, axis=1)

        axes[1].plot(range(len(means)), means, color=color, label=name)

        # IC
        axes[1].fill_between(
            range(len(stds)),
            means - stds,
            means + stds, 
            color = f'{color}80' # add alpha channel
        )
            
        # Names
        axes[1].set_xlabel('Generations')
        axes[1].set_ylabel('Avg. Activations')
        axes[1].set_title(f'Componentes aggregated activations')
        axes[1].grid(True)
        axes[1].legend()

    out_fp = path.join(out_dir, f'subset_optimization.png')
    logger.info(mess=f'Saving plot to {out_fp}')
    fig.savefig(out_fp)


def plot_subsetting_optimization(
    clusters_activations: Dict[int, Dict[int, List[Dict[str, float]]]],
    out_dir: str,
    logger: Logger = SilentLogger()
):

    # Auxiliary function
    def dict_list_to_array_dict(dict_list):
        ''' List of dicts with same keys to dict of lists stacked in arrays '''
    
        out = {}
        
        # Invert
        for d in dict_list:
            for key, value in d.items():
                out.setdefault(key, []).append(value)
        
        # Stack
        out = {k: np.array(v) for k, v in out.items()}
                
        return out

    # STYLE MACRO
    STYLE = {
            'Cluster optimized'      : ('#DC7633',  .0), 
            'External non optimized' : ('#3498DB', -.1), 
            'Cluster non optimized'  : ('#52BE80', +.1)
    }
    EDGE_COLOR = 'black'
    
    # Plot individually for each cluster
    for cluster_idx, units_activations in clusters_activations.items():
    
        # Plot both errors bars and violinplots
        erorrbars, ax1 = plt.subplots(figsize=(16, 8))
        violins,   ax2 = plt.subplots(figsize=(16, 8))
        
        # Unify information
        units_activations = {unit: dict_list_to_array_dict(values) for unit, values in units_activations.items()}
        
        # Lineplots points to be computed
        linesX = {k: [] for k in STYLE}
        linesY = {k: [] for k in STYLE}
        
        # Each point refers to a specific optimized units
        for unit_id, activations in units_activations.items():
            
            # Three components of the optimization
            for name, y in activations.items():
                
                # Styling associated to the component
                color, offset = STYLE[name]
                
                # ERROR BARS
                x = unit_id + offset * 0.2
                y_bar = np.mean(y)
                ax1.errorbar(x, y_bar, yerr=SEM(y), color=color, label=name, ecolor=color, fmt='o', markersize=8, capsize=10)
                
                linesX[name].append(x)
                linesY[name].append(y_bar)
                
                # VIOLIN PLOTS
                x = unit_id + offset * 0.5

                violin_parts = ax2.violinplot(
                    dataset=y,
                    positions=[x],
                    widths=0.1, 
                    showmeans=False,
                    showextrema=False
                )
                
                for pc in violin_parts['bodies']:  # type: ignore
                    pc.set_color(color)
                    pc.set_alpha(1)
                    pc.set_edgecolor(EDGE_COLOR)

        # Customizing legend
        legend_labels = [
            plt.Line2D(  # type: ignore
                [0], [0], color=color, lw=4, label=scr_type)
            for scr_type, (color, _) in STYLE.items()
        ]
        
        # Line plot for ERROR BARS
        for name, (color, _) in STYLE.items():
            ax1.plot(linesX[name], linesY[name], color=color, marker=',')
            
        # Axes names
        for ax in [ax1, ax2]:
            
            ax.legend(handles=legend_labels)
            ax.set_xlabel('Unit rank')
            ax.set_ylabel('Final activation')
            ax.set_title('Scoring Types comparison')
            ax.set_xticks(list(units_activations.keys()))

        # Save
        for fig, name in [
            (erorrbars, 'subset_optimization_error_bars'),
            (violins,   'subset_optimization_violin_plots'), 
        ]:
            out_fp = path.join(out_dir, f'cluster_{cluster_idx}-{name}.png')
            logger.info(mess=f'Saving plot to {out_fp}')
            fig.savefig(out_fp)

            
def plot_cluster_best_stimuli(
    cluster_codes: Dict[int, Dict[int, Tuple[float, Codes]]],
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    cluster_stimuli = {
    cluster_idx: [
            generator(
                codes=np.expand_dims(code, 0), # add batch size
            )[0] # first element of the tuple (the image)
            for _, code in codes.values()
        ]
        for cluster_idx, codes in cluster_codes.items()
    }

    cluster_grid = {
        cluster_idx: concatenate_images(
            img_list=stimuli, nrow = math.ceil(math.sqrt(len(stimuli)))
        )
        for cluster_idx, stimuli in cluster_stimuli.items()
    }
    
    for cluster_idx, grid in cluster_grid.items():
        
        out_fp = path.join(out_dir, f'cluster_{cluster_idx}-best_stimuli.png')
        logger.info(mess=f'Saving best cluster stimuli to {out_fp}')
        grid.save(out_fp)
        
def plot_cluster_target(
    cluster_codes: Dict[int, Dict[str, Tuple[float, Codes]]],
    generator: Generator,
    out_dir: str,
    logger: Logger = SilentLogger()
):
    
    cluster_stimuli = {
        cluster_idx: {
            'weighted' if weight_type else 'arithmetic':
            to_pil_image(
                generator(
                    codes=np.expand_dims(code, 0), # add batch size
                )[0] # first element of the tuple (the image)

            )
            for weight_type, (_, code) in codes.items()
        }
        for cluster_idx, codes in cluster_codes.items()
    }
    
    for cluster_idx, best_stimuli in cluster_stimuli.items():
        
        for weight_type, best_stimulus in best_stimuli.items():
            
            out_fp = path.join(out_dir, f'c{cluster_idx}_{weight_type}_mean-best_stimuli.png')
            logger.info(mess=f'Saving best cluster stimuli to {out_fp}')
            best_stimulus.save(out_fp)