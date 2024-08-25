
import os
from typing import Dict

import numpy as np
import seaborn as sns
from sklearn.metrics import mutual_info_score

from analysis.utils.misc import CurveFitter
from analysis.utils.misc import boxplots, end, load_clusters, load_imagenet, start
from analysis.utils.settings import CLU_ORDER, ALEXNET_DIR, COLORS, LAYER_SETTINGS, NEURON_SCALING_FN, OUT_DIR
from experiment.ClusterOptimization.plot import plot_clusters_superstimuli
from experiment.utils.args import WEIGHTS
from experiment.utils.misc import make_dir
from experiment.utils.settings import FILE_NAMES
from zdream.generator import DeePSiMGenerator
from zdream.utils.io_ import load_pickle
from zdream.utils.logger import LoguruLogger

# --- SETTINGS ---

LAYER = 'conv5-maxpool'
GEN_VARIANT = 'fc7'

PLOT = {
    "cardinality"           : True,
    "clu_optimization"      : True,
    "clu_optimization_norm" : True,
    "entropy"               : False
}

# --- RUN ---


def main():
    
    out_dir = os.path.join(OUT_DIR, "clustering_analysis", "plots", LAYER_SETTINGS[LAYER]['directory'])
    layer_dir = os.path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])
    clu_dir = os.path.join(layer_dir, 'clusters')
    if LAYER != 'fc8': PLOT['entropy'] = False

    # LOAD

    logger = LoguruLogger(on_file=False)
    
    clusters = load_clusters(dir=clu_dir, logger=logger)
    
    if PLOT['entropy']: 
        _, imagenet_superclass = load_imagenet(logger=logger)
    
    if PLOT['clu_optimization'] or PLOT['clu_optimization_norm']:
        clu_opt_file = os.path.join(layer_dir, 'superstimuli', 'clusters_superstimuli.pkl')
        logger.info(f'Loading clustering optimization from {clu_opt_file}')
        clusters_optimization = load_pickle(clu_opt_file)
        print(clusters_optimization.keys())

        clusters_optimization = {k: clusters_optimization[k] for k in sorted(clusters_optimization.keys(), key=lambda x: CLU_ORDER[x])}
    
    if PLOT['clu_optimization'] and PLOT['clu_optimization_norm']:
        logger.info('Loading generator')
        generator = DeePSiMGenerator(root=WEIGHTS, variant=GEN_VARIANT)
    
    
    if PLOT['clu_optimization_norm']:
        logger.info(f'Loading fitted neuron scaling function from {NEURON_SCALING_FN}')
        normalize_fun: CurveFitter = load_pickle(NEURON_SCALING_FN)[GEN_VARIANT][LAYER]
    
    plot_dir = make_dir(out_dir, logger=logger)
    
    # 1. CARDINALITY
    
    if 'cardinality' in PLOT:
    
        start(logger=logger, name="Cluster Cardinalities")
        
        clu_size = {name: [len(c) for c in clu] for name, clu in clusters.items()}  # type: ignore
        
        boxplots(
            data      = clu_size,
            ylabel    = 'Cardinality',
            title     = f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Cardinality",
            file_name = f'cluster_cardinality',
            out_dir   = plot_dir,
            logger    = logger
        )
        
        if LAYER == 'fc6-relu':
            
            clu_size['DBSCAN'] = list(sorted(clu_size['DBSCAN']))[:-1]
            
            boxplots(
                data      = clu_size,
                ylabel    = 'Cardinality',
                title     = 'Clustering Cardinality',
                file_name = f'cluster_cardinality_v2',
                out_dir   = plot_dir,
                logger    = logger
            )
        
        end(logger=logger)
        
    else: 
        
        logger.info(mess=f'Skipping Cardinality')
        logger.info(mess=f'')
    
    # 2. OPTIMIZATION
    
    if PLOT['clu_optimization'] or PLOT['clu_optimization_norm']:
    
        start(logger, 'CLUSTERING OPTIMIZATION')
        
        clu_opt_fitness = {
            clu_algo: [np.max(opt['fitness']) for opt in scores.values()]
            for clu_algo, scores in clusters_optimization.items()
        }
        
        if PLOT['clu_optimization']:
        
            boxplots(
                data=clu_opt_fitness,
                ylabel='Fitness',
                title=f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Superstimuli Optimization",
                file_name='clu_optimization',
                out_dir=plot_dir,
                logger=logger
            )
            
        if PLOT['clu_optimization_norm']:
    
            clu_opt_means_normalized = {
                clu_name: [mean / normalize_fun(len(clu)) for clu, mean in zip(clusters[clu_name], means)]  # type: ignore
                for clu_name, means in clu_opt_fitness.items()
            }
            
            boxplots(
                data=clu_opt_means_normalized,
                ylabel='Fitness',
                title=f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Superstimuli Optimization Normalized by Random Activation",
                file_name='clu_optimization_normalized',
                out_dir=plot_dir,
                logger=logger
            )
            
        if PLOT['clu_optimization'] and PLOT['clu_optimization_norm']:
            
            clu_opt_dir = make_dir(os.path.join(plot_dir, 'clu_optimization'), logger)
        
            for clu_algo, superstimuli in clusters_optimization.items():
                
                clu_algo_dir = make_dir(os.path.join(clu_opt_dir, clu_algo), logger)

                color = COLORS[CLU_ORDER[clu_algo]]

                palette =\
                    list(sns.dark_palette (color, n_colors=len(superstimuli)              ))[len(superstimuli)//2:] +\
                    list(sns.light_palette(color, n_colors=len(superstimuli), reverse=True))[:len(superstimuli)//2]
                
                plot_clusters_superstimuli(
                    superstimulus=superstimuli,
                    normalize_fun=normalize_fun,
                    clusters=clusters[clu_algo],
                    generator=generator,
                    out_dir=clu_algo_dir,
                    logger=logger,
                    PALETTE=palette
                )
    
        end(logger)

    else:
        
        logger.info(mess=f'Skipping Optimization')
        logger.info(mess=f'')
    
    # 3. ENTROPY
    
    if PLOT['entropy']:
    
        start(logger, 'CLUSTERING ENTROPY')
        
        entropies = {}
        
        for cluster_type, clusters_ in clusters.items():
            
            h = []
            
            for cluster in clusters_: # type: ignore
                inet_superclasses = [imagenet_superclass[object.label].name for object in cluster]
                h_ = mutual_info_score(inet_superclasses, inet_superclasses)
                h.append(h_)
                    
            entropies[cluster_type] = h
            
        boxplots(
            data      = entropies,
            ylabel    = 'Entropy',
            title     = 'Superclasses entropy',
            file_name = f'entropy',
            out_dir   = out_dir,
            logger    = logger
        )
        
        end(logger)
    
    else:
        
        logger.info(mess=f'Skipping Entropy')
        logger.info(mess=f'')
    
    logger.close()
    
    
if __name__ == '__main__':
    
    for layer in LAYER_SETTINGS.keys():
    
        LAYER = layer
        
        main()

