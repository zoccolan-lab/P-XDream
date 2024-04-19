
from typing import Any, List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec


NAME = 'opt_comparison_cluster'

# First cluster
# [470 503 504 505 572 618 619 626 631 666 696 710 711 720 725 737 773 804 813 828 846 849 855 859 868 883 898 899 907 910 966 968]

VALUES         = [0.1, 0.5, 1., 2., 5., 10., 20., 50.]
HYPERPARAMETER = 'sigma0'
TYPE           = 'cmaes'
UNITS          = '21=[1 2 3]'
ITER           = 2
SAMPLE         = 2


def get_arguments_opt_comparison(
    sample: int
) -> Tuple[str, str]:
        
        args = [
            ( opt, str(np.random.randint(1000, 100000)) )
            for opt in ['genetic', 'cmaes']
            for _   in range(sample)
        ]
        
        optimizer_str   = '#'.join([a for a, _ in args])
        random_seed_str = '#'.join([a for _, a in args])
        
        return optimizer_str, random_seed_str

def get_arguments_layer_comparison(
    values: List[Any],
    sample: int
) -> Tuple[str, str]:
    
        args = [
            (v, str(np.random.randint(1000, 100000)) )
            for v in values
            for _ in range(sample)
        ]
        
        values_str      = '#'.join([str(a) for a, _ in args])
        random_seed_str = '#'.join([a for _, a in args])
        
        return values_str, random_seed_str


if __name__ == '__main__':
    
    print('Multiple run: ')
    print('[1] Optimizers comparisons ')
    print('[2] Hyperparameter tuning comparisons ')
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            optimizer_str, random_seed_str = get_arguments_opt_comparison(
                sample=SAMPLE
            )
            
            args = {
                'iter'           : ITER,
                'name'           : NAME,
                'rec_layers'     : f'"{UNITS}"',
                'optimizer_type' : optimizer_str,
                'random_seed'    : random_seed_str,
            }
            
            file = 'run_multiple_optimizer_comparison.py'
            
        case 2:
            
            values_str, random_seed_str = get_arguments_layer_comparison(
                sample=SAMPLE,
                values=VALUES
            )
            
            args = {
                'iter'           : ITER,
                'name'           : NAME,
                'rec_layers'     : f'"{UNITS}"',
                HYPERPARAMETER   : values_str,
                'random_seed'    : random_seed_str,
                'optimizer_type' : TYPE,
                'hyperparameter' : HYPERPARAMETER
            }
            
            file = 'run_multiple_hyperparameter_tuning.py'

            
    copy_exec(
        file=file,
        args=args
    )
