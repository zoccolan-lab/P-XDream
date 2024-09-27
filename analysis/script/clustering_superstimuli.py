
import os
from typing import Any, Dict

import numpy as np
import seaborn as sns
from sklearn.metrics import mutual_info_score

from analysis.utils.misc import AlexNetLayerLoader, CurveFitter
from analysis.utils.misc import boxplots, end, load_imagenet, start
from analysis.utils.settings import CLU_ORDER, ALEXNET_DIR, COLORS, LAYER_SETTINGS, OUT_DIR
from experiment.ClusterOptimization.plot import generate_cluster_units_superstimuli, generate_clusters_superstimuli, plot_clusters_superstimuli, plot_dsweighting_clusters_superstimuli
from experiment.utils.args import WEIGHTS
from experiment.utils.misc import make_dir
from experiment.utils.settings import FILE_NAMES
from pxdream.generator import DeePSiMGenerator
from pxdream.utils.io_ import load_pickle
from pxdream.utils.logger import LoguruLogger

# --- SETTINGS ---

LAYER = 'fc8'
CLU_GEN_VARIANT = 'fc7'
UNITS_GEN_VARIANT = 'fc8'


# --- RUN ---

def main():
    
    out_dir   = os.path.join(OUT_DIR, "clustering_analysis", "superstimuli", LAYER_SETTINGS[LAYER]['directory'])
    layer_dir = os.path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])
    clu_dir   = os.path.join(layer_dir, 'clusters')

    # LOAD

    logger = LoguruLogger(to_file=False)
    
    layer_loader = AlexNetLayerLoader(alexnet_dir=ALEXNET_DIR, layer=LAYER, logger=logger)
    clusters = layer_loader.load_clusters()
    
    _, imagenet_superclass = load_imagenet(logger=logger)
    
    superstimuli_dir        = os.path.join(layer_dir, 'superstimuli')
    superstimuli_clu_file   = os.path.join(superstimuli_dir, 'cluster_superstimuli.pkl')
    superstimuli_units_file = os.path.join(superstimuli_dir, 'units_superstimuli.pkl')

    logger.info(f'Loading clustering optimization from {superstimuli_clu_file}')
    clu_superstimuli   = load_pickle(superstimuli_clu_file)
    units_superstimuli = load_pickle(superstimuli_units_file)

    dsw_superstimuli = clu_superstimuli.pop('DominantSetWeighted')
    dsw_superstimuli_clu = {'DominantSetWeighted': dsw_superstimuli}

    print(clu_superstimuli.keys())

    clu_superstimuli = {k: clu_superstimuli[k] for k in sorted(clu_superstimuli.keys(), key=lambda x: CLU_ORDER[x])}
    
    logger.info('Loading generator')
    clu_generator   = DeePSiMGenerator(root=WEIGHTS, variant=CLU_GEN_VARIANT)
    units_generator = DeePSiMGenerator(root=WEIGHTS, variant=UNITS_GEN_VARIANT)
    
    normalize_fun: CurveFitter = layer_loader.load_norm_fun(gen_variant=CLU_GEN_VARIANT)
    
    make_dir(out_dir, logger=logger)
    
    # 1. CARDINALITY
    
    start(logger=logger, name="Cluster Cardinalities")
    
    clu_size = {name: [1. * len(c) for c in clu] for name, clu in clusters.items()}  # type: ignore
    
    boxplots(
        data      = clu_size,
        ylabel    = 'Cardinality',
        title     = f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Cardinality",
        file_name = f'cluster_cardinality',
        out_dir   = out_dir,
        logger    = logger
    )
        
    end(logger)
    
    # 2. CLUSTER OPTIMIZATION STIMULI

    start(logger=logger, name="Cluster Optimization Stimuli")

    for clu_algo, superstimulus in (clu_superstimuli | dsw_superstimuli_clu).items():

        out_clu_algo_dir = make_dir(os.path.join(out_dir, clu_algo), logger=logger)

        color = COLORS[int(CLU_ORDER[clu_algo])]

        palette =\
            list(sns.dark_palette (color, n_colors=len(superstimulus)              ))[len(superstimulus)//2:] +\
            list(sns.light_palette(color, n_colors=len(superstimulus), reverse=True))[:len(superstimulus)//2]
        
        logger.info(f'Generating clusters superstimulus for {clu_algo}')

        generate_clusters_superstimuli(
            superstimulus=superstimulus,
            generator=clu_generator,
            out_dir=out_clu_algo_dir,
            logger=logger
        )
    
    end(logger)

    # 3. CLUSTER UNITS OPTIMIZATION STIMULI

    start(logger=logger, name="Cluster Units Optimization Stimuli")

    for clu_algo, clu in clusters.items():

        out_clu_algo_dir = make_dir(os.path.join(out_dir, clu_algo), logger=logger)
        
        logger.info(f'Generating  cluster units superstimulus for {clu_algo}')

        generate_cluster_units_superstimuli(
            superstimulus=units_superstimuli, # type: ignore
            clusters=clu,
            generator=units_generator,
            out_dir=out_clu_algo_dir,
            logger=logger
        )

    # 4. CLUSTER OPTIMIZATION PLOT

    start(logger=logger, name="Cluster Optimization Stimuli")

    for clu_algo, superstimulus in (clu_superstimuli | dsw_superstimuli_clu).items():

        out_clu_algo_dir = make_dir(os.path.join(out_dir, clu_algo), logger=logger)

        color = COLORS[int(CLU_ORDER[clu_algo])]

        palette =\
            list(sns.dark_palette (color, n_colors=len(superstimulus)              ))[len(superstimulus)//2:] +\
            list(sns.light_palette(color, n_colors=len(superstimulus), reverse=True))[:len(superstimulus)//2]
        
        logger.info(f'Generating clusters superstimulus for {clu_algo}')

        plot_clusters_superstimuli(
            superstimulus=superstimulus,
            normalize_fun=normalize_fun,
            clusters=clusters[clu_algo.replace('Weighted', '')],
            out_dir=out_clu_algo_dir,
            logger=logger,
            PALETTE=palette,
            TITLE=f'{LAYER_SETTINGS[LAYER]["title"]}, {clu_algo} Clustering'
        )
    
    end(logger)

    # 5. CLUSTER FITNESS COMPARISON

    start(logger=logger, name="Cluster Optimization Plot")

    clu_opt_fitness = {
        clu_algo: [np.max(opt['fitness']) for opt in scores.values()]
        for clu_algo, scores in clu_superstimuli.items()
    }
    
    boxplots(
        data=clu_opt_fitness,
        ylabel='Fitness',
        title=f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Superstimuli Optimization",
        file_name='clu_optimization',
        out_dir=out_dir,
        logger=logger
    )

    clu_opt_fitness = {
        clu_algo: [fitness / normalize_fun(x=len(clu)) for fitness, clu in zip(clu_fitness, clusters[clu_algo])]  # type: ignore
        for clu_algo, clu_fitness in clu_opt_fitness.items()
    }

    boxplots(
        data=clu_opt_fitness,
        ylabel='Rand-Normalized Fitness',
        title=f"{LAYER_SETTINGS[LAYER]['title']} - Clustering Superstimuli Optimization Normalized by Random Fitness",
        file_name='clu_optimization_normalized',
        out_dir=out_dir,
        logger=logger
    )

    # 6. CLUSTER DOMINANT SET WEIGHTING

    start(logger=logger, name="Cluster Fitness Comparison")

    ds_algo_name = 'DominantSet'
    ds_col       = COLORS[CLU_ORDER[ds_algo_name]]
    dsw_col      = "#656e6e"

    out_clu_algo_dir = make_dir(os.path.join(out_dir, ds_algo_name), logger=logger)

    ds_superstimuli = clu_superstimuli[ds_algo_name]

    arithmetic_palette = \
        list(sns.dark_palette (ds_col, n_colors=len(ds_superstimuli)              ))[len(ds_superstimuli)//2:] +\
        list(sns.light_palette(ds_col, n_colors=len(ds_superstimuli), reverse=True))[:len(ds_superstimuli)//2]
    
    weighted_palette = \
        list(sns.dark_palette (dsw_col, n_colors=len(dsw_superstimuli)              ))[len(dsw_superstimuli)//2:] +\
        list(sns.light_palette(dsw_col, n_colors=len(dsw_superstimuli), reverse=True))[:len(dsw_superstimuli)//2]
    
    plot_dsweighting_clusters_superstimuli(
        superstimuli= (ds_superstimuli, dsw_superstimuli),
        out_dir= out_clu_algo_dir,
        logger= logger,
        PALETTES=(arithmetic_palette, weighted_palette),
        TITLE=LAYER_SETTINGS[LAYER]["title"]
    )

    # 7. ENTROPY AND SUPERCLASSES COUNT

    start(logger, 'CLUSTERING ENTROPY')

    if LAYER == 'fc8':

        superlcasses_count = {}
        entropies = {}
        
        for cluster_type, clusters_ in clusters.items():
            
            h = []
            c = []
            
            for cluster in clusters_: # type: ignore
                inet_superclasses = [imagenet_superclass[object.label].name for object in cluster]
                c_ = len(set(inet_superclasses))
                h_ = mutual_info_score(inet_superclasses, inet_superclasses)
                h.append(h_)
                c.append(c_)
                    
            entropies         [cluster_type] = h
            superlcasses_count[cluster_type] = c
            
        boxplots(
            data      = entropies,
            ylabel    = 'Entropy',
            title     = f'{LAYER_SETTINGS[LAYER]["title"]} - Entropy with WordNet Superclasses',
            file_name = f'entropy',
            out_dir   = out_dir,
            logger    = logger,
        )

        boxplots(
            data      = superlcasses_count,
            ylabel    = 'Superclasses Count',
            title     = f'{LAYER_SETTINGS[LAYER]["title"]} - Superclasses Count',
            file_name = f'superclasses_count',
            out_dir   = out_dir,
            logger    = logger,
        )
        
        end(logger)

    logger.close()

    
if __name__ == '__main__':
    
    for layer in LAYER_SETTINGS.keys():
    
        LAYER = layer
        
        main()