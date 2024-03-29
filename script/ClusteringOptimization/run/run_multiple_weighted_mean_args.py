
from typing import List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec


NAME = 'scoring_comparison'

CLUSTER_IDX   = list(range(10))
ITER          = 200
SAMPLE        = 15

def get_arguments(
        cluster_idx:  List[int],
        sample: int
) -> Tuple[str, str, str]:
        
        tot_examples = len(cluster_idx) * sample * 2
        
        cluster_idx_str =    '#'.join([str(clu_idx) for clu_idx in cluster_idx for _ in range(sample)]*2)
        
        weighted_score_str = '#'.join(['' for _ in range(tot_examples//2)] + ['True' for _ in range(tot_examples//2)])
        weighted_score_str = f'"{weighted_score_str}"'
        
        random_seed_str =    '#'.join([str(np.random.randint(1000, 100000)) for _ in range(tot_examples)])
        

        return cluster_idx_str, weighted_score_str, random_seed_str


if __name__ == '__main__':
    
    cluster_idx_str, weighted_score_str, random_seed_str = get_arguments(
        cluster_idx=CLUSTER_IDX,
        sample=SAMPLE
    )


    args = {
        'iter'           : ITER,
        'name'           : NAME,
        'template'       : 'T',
        'cluster_idx'    : cluster_idx_str,
        'weighted_score' : weighted_score_str,
        'random_seed'    : random_seed_str,
    }

    copy_exec(
        file='run_multiple_weighted_mean.py',
        args=args
    )





