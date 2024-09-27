
from os import path
import os
from matplotlib import pyplot as plt

import seaborn as sns

from pxdream.clustering.ds import DSClusters
from pxdream.utils.logger import Logger, SilentLogger


def plot_cluster_extraction_trend(
    clusters: DSClusters,
    out_dir: str,
    exclude_singletons: bool = True,
    title: str = "",
    file_name: str = 'DS_trend.svg',
    logger: Logger = SilentLogger()
):
    
    PALETTE     = 'deep'
    FONT        = 'serif'
    FIG_SIZE    = (14, 8)
    TITLE_ARGS  = {'fontsize': 23, 'fontfamily': FONT}
    LEGEND_ARGS = {'prop': {'family': FONT, 'size': 14}}
    LABEL_ARGS  = {'fontsize': 18, 'fontfamily': FONT, 'labelpad': 15}  # fontsize: 18
    PLOT_ARGS   = {'markersize': 4, 'linewidth': 1.5} # {'markersize': 6, 'linewidth': 2.5}
    TICK_PARAMS = {'width': 1.5, 'length': 7, 'labelsize': 14}

    # Filter clusters and extract cardinality and coherence values
    clusters = [(len(c), c.W) for c in clusters if len(c) != 1 and exclude_singletons]  # type: ignore
    lens, Ws = zip(*clusters)
    x = list(range(1, len(clusters) + 1))

    # Set a color palette
    sns.set_palette(PALETTE)

    # Create a figure and Axes objects
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax2 = ax1.twinx()

    color1, color2 = (sns.color_palette()[i] for i in [0, 1])

    ax1.set_xlabel('Cluster ID',  **LABEL_ARGS)
    ax1.set_ylabel('Cardinality', **LABEL_ARGS)
    ax2.set_ylabel('Coherence',   **LABEL_ARGS)
    
    # Color the tick marks for the primary y-axis and make them thicker
    ax1.tick_params(colors='black', axis='x', color=color1, **TICK_PARAMS)
    ax1.tick_params(colors='black', axis='y', color=color1, **TICK_PARAMS)
    ax2.tick_params(colors='black', axis='y', color=color2, **TICK_PARAMS)    

    ax1.set_ylim(top=70)
    
    # Apply grid only to the primary y-axis
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Set zorder for grid to ensure it's below the lines
    ax1.grid(axis='y', linestyle='--', color=color1,  alpha=0.5)
    ax2.grid(axis='y', linestyle='--', color=color2,  alpha=0.5)  # Ensure grid lines are below the legend
    ax1.grid(axis='x', linestyle='--', color='black', alpha=0.5)

    # Plot the lines with default zorder
    line1, = ax1.plot(x, lens, marker='o', color=color1, linestyle='-',  label='Cardinality', **PLOT_ARGS)
    line2, = ax2.plot(x,   Ws, marker='s', color=color2, linestyle='--', label='Coherence',   **PLOT_ARGS)

    # Set a common title for the figure
    fig.suptitle(f'Iterative Extraction of Dominant Set Clusters - {title}', **TITLE_ARGS)

    # Create a single legend for both lines with a white background
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(
        lines, 
        labels, 
        loc="upper right", 
        facecolor='white',   # Set the background color to white
        edgecolor='black',   # Optionally, add a black border to the legend
        **LEGEND_ARGS
    )

    # Adjust layout for better spacing
    # fig.tight_layout(rect=(0, 0, 1, 0.95))
    # Save the figure
    out_fp = os.path.join(out_dir, file_name)
    logger.info(msg=f'Saving DS trend plot to {out_fp}')
    fig.savefig(out_fp, dpi=300)

    
    
def plot_cluster_extraction_ranks(
    clusters: DSClusters,
    out_dir: str,
    exclude_singletons: bool = True,
    title: str = "",
    file_name: str = 'DS_ranks.svg',
    logger: Logger = SilentLogger()
):
    # Styling parameters
    FONT        = 'serif'
    FIG_SIZE    = (30, 8)
    LABEL_SIZE  = 16
    TITLE_SIZE  = 22
    TICK_SIZE   = 14
    LABEL_PAD   = 10
    TITLE_PAD   = 1.
    GRID_COLOR  = '#888888'
    PALETTE     = 'PuBu_r'

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Example data
    ranks = [None] + [
        [obj.rank * 100 for obj in cluster]
        for cluster in clusters             # type: ignore
        if len(cluster) > 1 and exclude_singletons
    ]

    # Plotting boxplots
    sns.boxplot(data=ranks, ax=ax, palette=PALETTE)  # type: ignore

    # Set labels and title with customized font properties
    ax.set_xlabel('Cluster-ID', fontsize=LABEL_SIZE, fontfamily=FONT, labelpad=LABEL_PAD)
    ax.set_ylabel('Rank', fontsize=LABEL_SIZE, fontfamily=FONT, labelpad=LABEL_PAD)
    ax.set_title(F'Intracluster Ranks of Dominant Set Clusters - {title}', fontsize=TITLE_SIZE, fontfamily=FONT, pad=TITLE_PAD)
    
    # Customize ticks
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)

    ax.set_xlim(-1, len(ranks)+1)
    
    # Set grid style and color
    ax.grid(axis='y', linestyle='--', color=GRID_COLOR, alpha=0.7)
    
    # Customize x-ticks
    ax.set_xticks(list(range(20, len(ranks)))[::20])
    ax.set_xticklabels(ax.get_xticks(), fontsize=TICK_SIZE, fontfamily=FONT)

    # Save the figure
    out_fp = os.path.join(out_dir, file_name)
    logger.info(msg=f'Saving DS ranks plot to {out_fp}')
    fig.savefig(out_fp, dpi=300)