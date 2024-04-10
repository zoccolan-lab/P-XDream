
from typing import Any, List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec


NAME = 'opt_comparison_cluster'

# First cluster
# [470 503 504 505 572 618 619 626 631 666 696 710 711 720 725 737 773 804 813 828 846 849 855 859 868 883 898 899 907 910 966 968]

VALUES         = [0.1, 0.5, 1., 2., 5., 10., 20., 50.]
HYPERPARAMETER = 'sigma0'
TYPE           = 'cmaes'
UNITS          = "21=[470 503 504 505 572 618 619 626 631 666 696 710 711 720 725 737 773 804 813 828 846 849 855 859 868 883 898 899 907 910 966 968]"
ITER           = 200
SAMPLE         = 15


def get_arguments_opt_comparison(
    sample: int
) -> Tuple[str, str]:
        
        tot_examples =  sample * 2
        
        optimizer_str   = '#'.join([opt for opt in ['genetic', 'cmaes'] for _ in range(sample)])
        random_seed_str = '#'.join([str(np.random.randint(1000, 100000)) for _ in range(tot_examples)])
        
        return optimizer_str, random_seed_str

def get_arguments_layer_comparison(
    values: List[Any],
    sample: int
) -> Tuple[str, str]:
        
        tot_examples =  sample * len(values)
        
        values_str      = '#'.join([str(v) for v in values for _ in range(sample)])
        random_seed_str = '#'.join([str(np.random.randint(1000, 100000)) for _ in range(tot_examples)])
        
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
