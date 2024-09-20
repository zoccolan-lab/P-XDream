import math
from os import path
from typing import List, Tuple

import numpy as np

from experiment.utils.args import ExperimentArgParams
from zdream.utils.misc       import copy_exec
from zdream.utils.parameters import ArgParams

def get_rnd_seed() -> str:
    return str(np.random.randint(1000, 100000))

ITER         = '150'
TEMPLATE     = 'T'
GEN_VARIANT  = 'fc7'

NAME = f'prova'

SAMPLE = 2
ITER   = [100, 200]

def get_bmm_args():
    
    args = [
        (iter, get_rnd_seed())
        for iter in ITER
        for _ in range(SAMPLE)
    ]
    
    iter_args = "#".join([str(a) for a, _ in args])
    seed_args = "#".join([str(a) for _, a in args])
    
    return iter_args, seed_args
    
    

if __name__ == '__main__':
    
    args = {}
    
    print('Multiple run: ')
    
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            iter_args, seed_args = get_bmm_args()
            
            args[str(          ArgParams.NumIterations )] = iter_args
            args[str(          ArgParams.RandomSeed)]     = seed_args
            
            file = 'run_multi.py'
            
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
    
    args[str(          ArgParams.ExperimentName)] = NAME
    args[str(ExperimentArgParams.Template      )] = TEMPLATE
    args[str(ExperimentArgParams.GenVariant    )] = GEN_VARIANT

    copy_exec(file=file, args=args)
