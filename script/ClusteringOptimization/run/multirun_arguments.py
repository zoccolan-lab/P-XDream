
from typing import List, Tuple

import numpy as np

from script.ClusteringOptimization.parser import LOCAL_SETTINGS
from zdream.clustering.ds import DSClusters
from zdream.utils.io_ import read_json
from zdream.utils.misc import copy_exec

CLUSTER_FILE = read_json(LOCAL_SETTINGS)['clustering']

CLUSTER_CARDINALITY = {
    i: len(cluster)
    for i, cluster in enumerate(DSClusters.from_file(CLUSTER_FILE)) # type: ignore
}

NAME = 'subsetting_optimization_alexnetfc8_single_unit_3clusters'

CLUSTER_IDX   = list(range(3))
ITER          = 200
SAMPLE        = 12


def get_arguments_weighting(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        tot_examples = len(cluster_idx) * sample * 2
        
        cluster_idx_str =    '#'.join([str(clu_idx) for clu_idx in cluster_idx for _ in range(sample)]*2)
        
        weighted_score_str = '#'.join(['' for _ in range(tot_examples//2)] + ['True' for _ in range(tot_examples//2)])
        weighted_score_str = f'"{weighted_score_str}"'
        
        random_seed_str =    '#'.join([str(np.random.randint(1000, 100000)) for _ in range(tot_examples)])
        
        return cluster_idx_str, weighted_score_str, random_seed_str
    
    
def get_arguments_scoring(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        tot_examples = len(cluster_idx) * sample * 3
        
        cluster_idx_str =    '#'.join([str(clu_idx) for clu_idx in cluster_idx for _ in range(sample)]*3)
        
        weighted_score_str = '#'.join(
            ['cluster'    for _ in range(tot_examples//3)] +\
            ['random'     for _ in range(tot_examples//3)] +\
            ['random_adj' for _ in range(tot_examples//3)]
        )
        weighted_score_str = f'"{weighted_score_str}"'
        
        
        random_seed_str =    '#'.join([str(np.random.randint(1000, 100000)) for _ in range(tot_examples)])
        

        return cluster_idx_str, weighted_score_str, random_seed_str
    
def get_arguments_subset_optimization(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        triples = [ 
                (
                    opt_unit+1,
                    clu_idx,
                    str(np.random.randint(1000, 100000))
                )
                for _ in range(sample)
                for clu_idx in cluster_idx 
                for opt_unit in range(CLUSTER_CARDINALITY[clu_idx])
        ]
        
        opt_units_str = '#'.join([str(a) for a, _, _ in triples])
        clu_idx_str   = '#'.join([str(a) for _, a, _ in triples])
        rnd_seed_str  = '#'.join([str(a) for _, _, a in triples])
        
        return clu_idx_str, opt_units_str, rnd_seed_str


if __name__ == '__main__':
    
    choices = {
        "weighting": {
            'fun'  : get_arguments_weighting,
            'arg'  : 'weighted_score',
            'file' : 'run_multiple_weighted_mean.py'
        },
        'scoring': {
            'fun'  : get_arguments_scoring,
            'arg'  : 'scr_type',
            'file' : 'run_multiple_scr_type.py'
        },
        'subsetting': {
            'fun'  : get_arguments_subset_optimization,
            'arg'  : 'opt_units',
            'file' : 'run_multiple_subset_optimization.py'
        },
        'subsetting_topk': {
            'fun'  : get_arguments_subset_optimization,
            'arg'  : 'opt_units',
            'file' : 'run_multiple_subset_optimization.py'
        },
        'subsetting_botk': {
            'fun'  : get_arguments_subset_optimization,
            'arg'  : 'opt_units',
            'file' : 'run_multiple_subset_optimization.py'
        }
    }
    
    print('Multiple run: ')
    print('[1] Weighting ')
    print('[2] Scoring type ')
    print('[3] Subset optimization single unit')
    print('[4] Subset optimization top-k unit')
    print('[5] Subset optimization bottom-k unit')
    choice = int(input('Choice: '))
    
    values = list(choices.values())[choice-1]
    
    cluster_idx_str, arg_str, random_seed_str = values['fun'](
        cluster_idx=CLUSTER_IDX,
        sample=SAMPLE
    )
            
    args = {
        'iter'           : ITER,
        'name'           : NAME,
        'template'       : 'T',
        'cluster_idx'    : cluster_idx_str,
        values['arg']    : arg_str,
        'random_seed'    : random_seed_str,
    }
    
    if choice == 3:
        args['scr_type'] = 'subset'
    if choice == 4:
        args['scr_type'] = 'subset_top'
    if choice == 5:
        args['scr_type'] = 'subset_bot'
    
    copy_exec(
        file=values['file'],
        args=args
    )






