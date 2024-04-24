import math
from typing import List, Tuple

import numpy as np

from zdream.clustering.ds import DSClusters
from zdream.utils.misc import copy_exec


CLUSTER_FILE  = '/data/Zdream/clustering/alexnet/conv5-maxpool/DSClusters.json'

NAME = 'subset_optimization_alexnetconv5maxpool_singleunit_c0'

CLUSTER_CARDINALITY = {
    i: len(cluster)
    for i, cluster in enumerate(DSClusters.from_file(CLUSTER_FILE)) # type: ignore
}

CLUSTER_IDX   = [0]

ITER          = 150
SAMPLE        = 1
LAYER         = 13


def get_arguments_weighting(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        args = [
            ( str(clu_idx), 'True' if weighted else '', str(np.random.randint(1000, 100000)) )
            for clu_idx in cluster_idx
            for weighted in [True, False]
            for _ in range(sample)
        ]
        
        cluster_idx_str =    '#'.join([a for a, _, _ in args])
        weighted_score_str = '#'.join([a for _, a, _ in args])
        random_seed_str =    '#'.join([a for _, _, a in args])
        
        return cluster_idx_str, weighted_score_str, random_seed_str
    
    
def get_arguments_scoring(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        args = [
            ( str(clu_idx), scr_type, str(np.random.randint(1000, 100000)) )
            for clu_idx in cluster_idx
            for scr_type in ['cluster', 'random', 'random_adj']
            for _ in range(sample)
        ]
        
        cluster_idx_str    = '#'.join([a for a, _, _ in args])
        weighted_score_str = '#'.join([a for _, a, _ in args])
        random_seed_str    = '#'.join([a for _, _, a in args])
        

        return cluster_idx_str, weighted_score_str, random_seed_str
    
def get_arguments_single_unit(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        args = [ 
                (
                    opt_unit+1,
                    clu_idx,
                    str(np.random.randint(1000, 100000))
                )
                for clu_idx in cluster_idx 
                for opt_unit in range(CLUSTER_CARDINALITY[clu_idx])
                for _ in range(sample)
        ]
        
        opt_units_str = '#'.join([str(a) for a, _, _ in args])
        clu_idx_str   = '#'.join([str(a) for _, a, _ in args])
        rnd_seed_str  = '#'.join([str(a) for _, _, a in args])
        
        return clu_idx_str, opt_units_str, rnd_seed_str
    
def get_arguments_topk_botk(
    cluster_idx:  List[int]
) -> List[Tuple[str, str]]:
    
    args = [
        (clu_idx, topbot, k)
        for clu_idx in cluster_idx
        for k in [
            1, 
            int(math.sqrt(CLUSTER_CARDINALITY[clu_idx])),
            CLUSTER_CARDINALITY[clu_idx]//2
        ]
        for topbot in ['subset_top', 'subset_bot']
    ]
    
    clu_idx_str = '#'.join([str(a) for a, _, _ in args])
    topbot_str  = '#'.join([str(a) for _, a, _ in args])
    k_str       = '#'.join([str(a) for _, _, a in args])
    
    return clu_idx_str, topbot_str, k_str

def get_arguments_best_stimuli(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        args = [ 
                (
                    opt_unit+1,
                    clu_idx,
                    str(np.random.randint(1000, 100000))
                )
                for clu_idx in cluster_idx 
                for opt_unit in range(CLUSTER_CARDINALITY[clu_idx])
                for _ in range(sample)
        ]
        
        opt_units_str = '#'.join([str(a) for a, _, _ in args])
        clu_idx_str   = '#'.join([str(a) for _, a, _ in args])
        rnd_seed_str  = '#'.join([str(a) for _, _, a in args])
        
        return clu_idx_str, opt_units_str, rnd_seed_str

def get_arguments_cluster_target(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        args = [ 
                (
                    clu_idx,
                    'True' if weight else '',
                    str(np.random.randint(1000, 100000))
                )
                for weight  in [True, False]
                for clu_idx in cluster_idx 
                for _ in range(sample)
        ]
        
        clu_idx_str      = '#'.join([str(a) for a, _, _ in args])
        weighted_scr_str = '#'.join([str(a) for _, a, _ in args])
        rnd_seed_str     = '#'.join([str(a) for _, _, a in args])
        
        return clu_idx_str, weighted_scr_str, rnd_seed_str


if __name__ == '__main__':
    
    print('Multiple run: ')
    print('[1] Weighting ')
    print('[2] Scoring type ')
    print('[3] Subset optimization single unit')
    print('[4] Subset optimization top-k, bot-k unit')
    print('[5] Best stimuli')
    print('[6] Cluster target stimuli')
    
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            clu_idx_str, weighted_score_str, rnd_seed_str = get_arguments_weighting(cluster_idx=CLUSTER_IDX, sample=SAMPLE)
            
            args = {
                'cluster_idx'    : clu_idx_str,
                'weighted_score' : weighted_score_str,
                'random_seed'    : rnd_seed_str,
                'scr_type'       : 'cluster'
            }
            
            file = 'run_multiple_weighted_mean.py'
            
        case 2:
            
            clu_idx_str, scr_type_str, rnd_seed_str = get_arguments_scoring(cluster_idx=CLUSTER_IDX, sample=SAMPLE)
            
            args = {
                'cluster_idx'    : clu_idx_str,
                'scr_type'       : scr_type_str,
                'random_seed'    : rnd_seed_str,
            }
            
            file = 'run_multiple_scr_type.py'
            
        case 3:
            
            clu_idx_str, opt_units_str, rnd_seed_str = get_arguments_single_unit(cluster_idx=CLUSTER_IDX, sample=SAMPLE)
            
            args = {
                'cluster_idx'    : clu_idx_str,
                'opt_units'      : opt_units_str,
                'random_seed'    : rnd_seed_str,
                'scr_type'       : 'subset'
            }
            
            file = 'run_multiple_subset_optimization.py'
            
        case 4:
            
            clu_idx_str, topbot_str, k_str = get_arguments_topk_botk(cluster_idx=CLUSTER_IDX)
            
            args = {
                'cluster_idx'    : clu_idx_str,
                'opt_units'      : k_str,
                'scr_type'       : topbot_str
            }
            
            file = 'run_multiple_subset_optimization.py'
            
        case 5:
            
            clu_idx_str, opt_units_str, rnd_seed_str = get_arguments_best_stimuli(cluster_idx=CLUSTER_IDX, sample=SAMPLE)
            
            args = {
                'cluster_idx'    : clu_idx_str,
                'opt_units'      : opt_units_str,
                'random_seed'    : rnd_seed_str,
                'scoring_type'   : 'subset'
            }
            
            file = 'run_multiple_best_stimuli.py'
            
        case 6:
            
            cluster_idx_str, weighted_score_str, rnd_seed_str = get_arguments_cluster_target(cluster_idx=CLUSTER_IDX, sample=SAMPLE)
            
            args = {
                'cluster_idx'    : cluster_idx_str,
                'weighted_score' : weighted_score_str,
                'random_seed'    : rnd_seed_str,
                'scr_type'       : 'cluster'
            }
            
            file = 'run_multiple_cluster_target.py'
            
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
            
    args['iter']         = ITER
    args['name']         = NAME
    args['template']     = 'T'
    args['layer']        = LAYER
    args['cluster_file'] = CLUSTER_FILE
    
    copy_exec(
        file=file,
        args=args
    )






