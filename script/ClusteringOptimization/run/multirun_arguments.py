
from typing import List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec


NAME = 'scoring_comparison'

CLUSTER_IDX   = list(range(2))
ITER          = 2
SAMPLE        = 2

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
    
        pass
        
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
        }
    }
    
    print('Multiple run: ')
    print('[1] Weighting ')
    print('[2] Scoring type ')
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
            
    copy_exec(
        file=values['file'],
        args=args
    )






