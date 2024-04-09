
from typing import Any, List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec


NAME = 'before_berha'

VALUES         = [2, 3, 4]
HYPERPARAMETER = 'n_parents'
TYPE           = 'genetic'
UNITS          = "21=[0:20]"
ITER           = 5
SAMPLE         = 3

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
                'rec_layers'     : UNITS,
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
                'rec_layers'     : UNITS,
                HYPERPARAMETER   : values_str,
                'random_seed'    : random_seed_str,
                'hyperparameter' : HYPERPARAMETER
            }
            
            file = 'run_multiple_hyperparameter_tuning.py'

            
    copy_exec(
        file=file,
        args=args
    )
