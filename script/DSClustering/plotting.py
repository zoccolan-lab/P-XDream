
from os import path
from matplotlib import pyplot as plt

import seaborn as sns

from zdream.clustering.ds import DSClusters
from zdream.logger import Logger, MutedLogger


def plot_cluster_extraction_trend(
    clusters: DSClusters,
    out_dir: str,
    logger: Logger = MutedLogger()
):
    
    # Extract clusters cardinalities
    lens = [len(c) for c in clusters] # type: ignore
    Ws   = [c.W    for c in clusters] # type: ignore
    
    # Create a figure and an Axes object
    fig, axes = plt.subplots(ncols=2, figsize=(15, 8))

    x = list(range(len(clusters)))

    for ax, y, ylab, col in zip(
        axes, 
        [lens, Ws],
        ['cardinality', 'coherence'], 
        ['#7FB3D5', '#EB984E']
    ):
        
        ax.plot(x, y, '-', color=col)
        
        ax.set_xlabel('Cluster-ID')
        ax.set_ylabel(f'Cluster {ylab}')

        ax.set_xticks(x[::4])
        ax.set_title('', fontsize=1)
        

    # Set common title
    fig.suptitle(f'DS Clusters iterative extraction', fontsize=16)
    
    # Save
    out_fp = path.join(out_dir, f'DS_trend.png')
    logger.info(mess=f'Saving DS trend plot to {out_fp}')
    fig.savefig(out_fp)
    
    
def plot_cluster_ranks(
    clusters: DSClusters,
    out_dir: str,
    logger: Logger = MutedLogger()
):
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x = list(range(1, len(clusters)+1))

    # Example data
    ranks = [
        [obj.rank * 100 for obj in cluster]
        for cluster in clusters             # type: ignore
    ]

    # Plotting violin plots
    sns.boxplot(data=ranks, ax=ax, palette='PuBu')

    # Set labels and title
    ax.set_xlabel('Cluster-ID')
    ax.set_ylabel('Rank')
    ax.set_title ('Cluster ranks')

    ax.set_xticks(list(range(len(clusters))))[::5]
    
    # Save
    out_fp = path.join(out_dir, f'DS_ranks.png')
    logger.info(mess=f'Saving DS ranks plot to {out_fp}')
    fig.savefig(out_fp)