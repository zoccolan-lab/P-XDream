'''
This script computes the hierarchy distance of the elements in the clusters of the different clusterings.
'''

import os
from typing import Dict, List, Tuple

import numpy as np
from   numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score

from zdream.clustering.cluster import Cluster, Clusters
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger
from analysis.utils.misc import start, end
from analysis.utils.settings import FILE_NAMES, LAYER_SETTINGS, OUT_DIR, OUT_NAMES, WORDNET_DIR
from analysis.utils.wordnet import ImageNetWords, WordNet

# ------------------------------------------- SETTINGS ---------------------------------------

LAYER       = 'fc8'
FIGSIZE     = (10, 8)
SNS_PALETTE = 'Set2'

_, NAME, _, _ = LAYER_SETTINGS[LAYER]

OUT_NAME = f'{OUT_NAMES["cluster_type_comparison"]}_{NAME}'
out_dir = os.path.join(OUT_DIR, OUT_NAME)

# ---------------------------------------------------------------------------------------------

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
    
    
def relative_mean_hierarchy_distance(
    id_      : int, 
    cluster      : Cluster,
    inet         : ImageNetWords,
    logger       : Logger = SilentLogger()
) -> Tuple[float, float] | None:
    '''
    Compute the mean hierarchy distance of an object to the other elements in the cluster

    :param id_: Object index
    :type id: int
    :param cluster: Cluster with its label
    :type cluster: Cluster
    :param inet: ImageNetWords object
    :type inet: ImageNetWords
    :param logger: Logger to log progress, defaults to SilentLogger()
    :type logger: Logger, optional
    :return: Mean hierarchy distance to the cluster and mean hierarchy distance w.r.t. the rest of the elements
    :rtype: Tuple[float, float] | None
    '''
    
    # Skip singletons
    if len(cluster) == 1:
        logger.warn(mess=f"Skipping object {id_}: it's a singleton")
        return
    
    assert id_ in cluster.labels
    
    # Compute mean dist to other elements in the cluster
    clu_mean = np.mean([
        wordnet.common_ancestor_distance(inet[id_], inet[clu_element.label])
        for clu_element in cluster if clu_element != id_  # type: ignore
    ])
    
    # Compute mean dist to other elements not in the cluster
    non_cluster = list(set([obj.id for obj in inet]).difference(cluster.labels))  # type: ignore
    
    non_clu_mean = np.mean([
        wordnet.common_ancestor_distance(inet[id_], inet[non_clu_element])
        for non_clu_element in non_cluster
    ])
    
    return  float(clu_mean),\
            float(non_clu_mean - clu_mean)


if __name__ == '__main__':
    
    # Create logger
    logger = LoguruLogger(on_file=False)
    
    # WordNet paths
    words_fp             = os.path.join(WORDNET_DIR, FILE_NAMES['words'])
    hierarchy_fp         = os.path.join(WORDNET_DIR, FILE_NAMES['hierarchy'])
    words_precomputed_fp = os.path.join(WORDNET_DIR, FILE_NAMES['words_precoputed'])
    
    # Load WordNet with precomputed words if available
    if os.path.exists(words_precomputed_fp):
        
        logger.info(mess='Loading precomputed WordNet')
        
        wordnet = WordNet.from_precomputed(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp, 
            words_precomputed=words_precomputed_fp,
            logger=logger
        )
    
    else:
        
        logger.info(mess=f'No precomputation found at {words_precomputed_fp}. Loading WordNet from scratch')
        
        wordnet = WordNet(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp,
            logger=logger
        )

        # Dump precomputed words for future use
        wordnet.dump_words(fp=WORDNET_DIR)

    # Load ImageNet
    logger.info(mess='Loading ImageNet')
    inet_fp = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet'])
    inet = ImageNetWords(imagenet_fp=inet_fp, wordnet=wordnet)
    
    # Load SuperImageNet
    logger.info(mess='Loading ImageNet superclasses')
    inet_super_fp = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet_super'])
    inet_super = ImageNetWords(imagenet_fp=inet_super_fp, wordnet=wordnet)
    
    # Loading Labelings
    labelings_fp = os.path.join(out_dir, FILE_NAMES['labelings'])
    logger.info(mess=f'Loading Labelings from {labelings_fp}')
    labelings: Dict[str, NDArray] = dict(np.load(labelings_fp))
    logger.info(f'Loaded labelings: {", ".join(labelings.keys())}')
    
    # Compute clustering
    clusters: Dict[str, Clusters] = {
        clu_name: Clusters.from_labeling(labeling=labeling)
        for clu_name, labeling in labelings.items()
    }
    
    # Create output directory
    hierarchy_dir = os.path.join(out_dir, 'hierarchy')
    logger.info(mess=f'Creating output directory to {hierarchy_dir}')
    os.makedirs(hierarchy_dir, exist_ok=True)
    
    # 1. LENGTHS
    
    start(logger, 'CLUSTERING CARDINALITY')
    
    clu_size: Dict[str, List[int]]= {
        clu_name: [len(cluster) for cluster in clusters]  # type: ignore
        for clu_name, clusters in clusters.items()
    }
    
    plot(
        data      = clu_size,
        ylabel    = 'Cardinality',
        title     = 'Clustering Cardinality',
        file_name = f'clu_size',
        out_dir   = hierarchy_dir,
        logger    = logger
    )
    
    end(logger)
    
    # 2. HIERARCHY DISTANCES
    
    start(logger, 'CLUSTERING HIERARCHY DISTANCES')
    
    hierarchy_dist     = {}
    hierarchy_dist_rel = {}

    for cluster_type, clusters_ in clusters.items():
        logger.info(f"Computing cluster type: {cluster_type}")
        dists    = []
        dists_rel = []
        
        for cluster in clusters_: # type: ignore
            
            for object in cluster:
                
                dists_out = relative_mean_hierarchy_distance(
                    id_=object.label, 
                    cluster=cluster,
                    inet=inet,
                    logger=logger
                )
                
                if dists_out is None: continue
                
                dist, dist_rel = dists_out
                
                dists    .append(dist)
                dists_rel.append(dist_rel)
                
        hierarchy_dist    [cluster_type] = dists
        hierarchy_dist_rel[cluster_type] = dists_rel
        
    plot(
        data      = hierarchy_dist,
        ylabel    = 'Hierarchy Distance',
        title     = 'Mean Hierarchy Distance',
        file_name = f'hierarchy_dist',
        out_dir   = hierarchy_dir,
        logger    = logger
    )
    
    plot(
        data      = hierarchy_dist_rel,
        ylabel    = 'Hierarchy Relative Distance',
        title     = 'Relative Mean Hierarchy Distance',
        file_name = f'hierarchy_dist_rel',
        out_dir   = hierarchy_dir,
        logger    = logger
    )
    
    end(logger)
    
    # 3. ENTROPY
    
    start(logger, 'CLUSTERING ENTROPY')
    
    entropies = {}
    
    for cluster_type, clusters_ in clusters.items():
        
        h = []
        
        for cluster in clusters_: # type: ignore
            inet_superclasses = [inet_super[object.label].name for object in cluster]
            h_ = mutual_info_score(inet_superclasses, inet_superclasses)
            h.append(h_)
                
        entropies[cluster_type] = h
        
    plot(
        data      = entropies,
        ylabel    = 'Entropy',
        title     = 'Entropy',
        file_name = f'entropy',
        out_dir   = hierarchy_dir,
        logger    = logger
    )
    
    end(logger)
    
    logger.close()