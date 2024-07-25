
import os
from typing import Dict

import numpy as np
from sklearn.metrics import mutual_info_score

from analysis.utils.misc import CurveFitter
from analysis.utils.misc import box_violin_plot, end, load_clusters, load_imagenet, start
from analysis.utils.settings import CLUSTER_DIR, LAYER_SETTINGS, NEURON_SCALING_FUN, OUT_DIR
from experiment.utils.misc import make_dir
from zdream.utils.io_ import load_pickle
from zdream.utils.logger import LoguruLogger

# --- SETTINGS ---

LAYER = 'conv5-maxpool'

out_dir = os.path.join(OUT_DIR, "clustering_analysis", "plots", LAYER_SETTINGS[LAYER]['directory'])
clu_dir = os.path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])

PLOT = {
    "cardinality"           : True,
    "clu_optimization"      : True,
    "clu_optimization_norm" : True,
    "entropy"               : True
}

if LAYER != 'fc8': PLOT['entropy'] = False
if LAYER == 'conv5-maxpool': PLOT['clu_optimization'] = PLOT['clu_optimization_norm'] = False


# --- RUN ---


def main():
    
    # LOAD

    logger = LoguruLogger(on_file=False)
    
    clusters = load_clusters(dir=clu_dir, logger=logger)
    
    if PLOT['entropy']: 
        _, imagenet_superclass = load_imagenet(logger=logger)
    
    if PLOT['clu_optimization'] or PLOT['clu_optimization_norm']:
        clu_opt_file = os.path.join(clu_dir, 'cluster_optimization.pkl')
        logger.info(f'Loading clustering optimization from {clu_opt_file}')
        clusters_optimization = load_pickle(clu_opt_file)
    
    if PLOT['clu_optimization_norm']:
        logger.info(f'Loading fitted neuron scaling function from {NEURON_SCALING_FUN}')
        normalize_fun: Dict[str, CurveFitter] = load_pickle(NEURON_SCALING_FUN)[LAYER]
    
    plot_dir = make_dir(out_dir, logger=logger)
    
    # 1. CARDINALITY
    
    if 'cardinality' in PLOT:
    
        start(logger=logger, name="Cluster Cardinalities")
        
        clu_size = {name: [len(c) for c in clu] for name, clu in clusters.items()}  # type: ignore
        
        box_violin_plot(
            data      = clu_size,
            ylabel    = 'Cardinality',
            title     = 'Clustering Cardinality',
            file_name = f'cluster_cardinality',
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
        
        clu_opt_means = {
            clu_name: [np.mean(opt) for opt in scores.values()]
            for clu_name, scores in clusters_optimization.items()
        }
        
        if PLOT['clu_optimization']:
        
            box_violin_plot(
                data=clu_opt_means,
                ylabel='Fitness',
                title='Clustering Optimization',
                file_name='clu_optimization',
                out_dir=plot_dir,
                logger=logger
            )
        
        if PLOT['clu_optimization_norm']:
    
            clu_opt_means_normalized = {
                clu_name: [mean / normalize_fun(len(clu)) for clu, mean in zip(clusters[clu_name], means) if len(clu) > 20]  # type: ignore
                for clu_name, means in clu_opt_means.items()
            }
            
            box_violin_plot(
                data=clu_opt_means_normalized,
                ylabel='Fitness',
                title='Clustering Optimization Normalized by Random Activation',
                file_name='clu_optimization_normalized',
                out_dir=plot_dir,
                logger=logger
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
            
        box_violin_plot(
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
    
    main()
    #for LAYER in LAYER_SETTINGS.keys():
    #    
    #    out_dir = os.path.join(OUT_DIR, "clustering_analysis", "plots", LAYER_SETTINGS[LAYER]['directory'])
    #    clu_dir = os.path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])
    #    
    #    if LAYER != 'fc8': PLOT['entropy'] = False
    #    if LAYER == 'conv5-maxpool': PLOT['clu_optimization'] = PLOT['clu_optimization_norm'] = False
        

