
from functools import partial
from typing import List, Tuple

import numpy as np

from script.cmdline_args import LOCAL_SETTINGS
from zdream.clustering.ds import DSClusters
from zdream.utils.io_ import read_json
from zdream.utils.misc import copy_exec

CLUSTER_FILE = read_json(LOCAL_SETTINGS)['clustering']

CLUSTER_CARDINALITY = {
    i: len(cluster)
    for i, cluster in enumerate(DSClusters.from_file(CLUSTER_FILE)) # type: ignore
}

NAME = 'trial'

CLUSTER_IDX   = list(range(2))
ITER          = 2
SAMPLE        = 2


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
    
def get_arguments_subset_optimization(
        cluster_idx:  List[int],
        sample: int,
        last: bool = True
) -> Tuple[str, str, str]:
        
        args = [ 
                (
                    opt_unit+1,
                    clu_idx,
                    str(np.random.randint(1000, 100000))
                )
                for clu_idx in cluster_idx 
                for opt_unit in range(CLUSTER_CARDINALITY[clu_idx] + (0 if last else -1))
                for _ in range(sample)
        ]
        
        opt_units_str = '#'.join([str(a) for a, _, _ in args])
        clu_idx_str   = '#'.join([str(a) for _, a, _ in args])
        rnd_seed_str  = '#'.join([str(a) for _, _, a in args])
        
        return clu_idx_str, opt_units_str, rnd_seed_str

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
            'fun'  : partial(get_arguments_subset_optimization, last=False),
            'arg'  : 'opt_units',
            'file' : 'run_multiple_subset_optimization.py'
        },
        'subsetting_botk': {
            'fun'  : partial(get_arguments_subset_optimization, last=False),
            'arg'  : 'opt_units',
            'file' : 'run_multiple_subset_optimization.py'
        },
        'best_stimuli': {
            'fun'  : get_arguments_best_stimuli,
            'arg'  : 'opt_units',
            'file' : 'run_multiple_best_stimuli.py'
        }
    }
    
    print('Multiple run: ')
    print('[1] Weighting ')
    print('[2] Scoring type ')
    print('[3] Subset optimization single unit')
    print('[4] Subset optimization top-k unit')
    print('[5] Subset optimization bottom-k unit')
    print('[6] Best stimuli')
    
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
    
    if choice == 3 or choice == 6:
        args['scr_type'] = 'subset'
    if choice == 4:
        args['scr_type'] = 'subset_top'
    if choice == 5:
        args['scr_type'] = 'subset_bot'
    
    copy_exec(
        file=values['file'],
        args=args
    )






