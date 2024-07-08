
import random
from typing import List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec

def generate_log_numbers(N, M): return list(sorted(list(set([int(a) for a in np.logspace(0, np.log10(M), N)]))))

IDX_SAMPLE = 145

NAME = f'sample_activity_maximization_neuron{IDX_SAMPLE}_250iter'

ITER     =  250
SAMPLE   =   30
N_POINTS =   45

# Layer : Neurons
LAYERS = [
    (13, 9216), # Conv5 MaxPool
    (17, 4096), # FC6 Relu
    (20, 4096), # FC7 Relu
    (21, 1000)  # FC8 Maxpool
]

SAMPLE_REC_LAYER = f"21=[{IDX_SAMPLE}]"
SAMPLE_SCR_LAYER = "21=[]"


def neuron_scaling_args() -> Tuple[str, str, str]:

    args = [
        (f'{layer}={neuron}r[]', f'{layer}=[]', str(random.randint(1000, 1000000)))
        for layer, neurons in LAYERS
        for neuron in generate_log_numbers(N_POINTS, neurons)
        for _ in range(SAMPLE)
    ]

    rec_layer_str = '#'.join(a for a, _, _ in args)
    scr_layer_str = '#'.join(a for _, a, _ in args)
    rand_seed_str = '#'.join(a for _, _, a in args)
    
    return rec_layer_str, scr_layer_str, rand_seed_str


def layers_correlation_arg() -> Tuple[str, str, str]:

    rec_layers_str = ','.join([f'{layer}=[]' for layer in LAYERS])
    
    args = [
        ( f'{layer}={neuron}r[]', str(random.randint(1000, 1000000)) )
        for layer, neu_len in LAYERS
        for neuron in generate_log_numbers(N_POINTS, neu_len)
        for _ in range(SAMPLE)]
    
    scr_layers_str = '#'.join(a for a, _ in args)
    rand_seed_str  = '#'.join(a for _, a in args)
    
    return rec_layers_str, scr_layers_str, rand_seed_str


def generator_variants() -> Tuple[str, str]:

    args = [
        ( f'{variant}', str(random.randint(1000, 1000000)) )
        for variant in VARIANTS
        for _ in range(SAMPLE)
    ]
    
    variant_str    = '#'.join(a for a, _ in args)
    rand_seed_str  = '#'.join(a for _, a in args)
    
    return variant_str, rand_seed_str


if __name__ == '__main__':

    print('Multiple run: ')
    print('[1] Neural scaling')
    print('[2] Layers correlation')
    print('[3] Maximize samples')
    print('[4] Generator variants')
    
    option = int(input('Choose option: '))
    
    match option:
        
        case 1:
            
            rec_layer_str, scr_layer_str, rnd_seed_str = neuron_scaling_args()
            
            args = {
                'rec_layers'  : rec_layer_str,
                'scr_layers'  : scr_layer_str,
                'random_seed' : rnd_seed_str,
            }
            
            file = 'run_multiple_neuronal_scaling.py'
            
        case 2:
            
            rec_layer_str, scr_layer_str, rnd_seed_str = layers_correlation_arg()
            
            args = {
                'rec_layers'  : rec_layer_str,
                'scr_layers'  : scr_layer_str,
                'random_seed' : rnd_seed_str,
            }
            
            file = 'run_multiple_layer_correlation.py'
            
        case 3:
                        
            args = {
                'rec_layers'  : SAMPLE_REC_LAYER,
                'scr_layers'  : SAMPLE_SCR_LAYER,
                'random_seed' : '#'.join(str(random.randint(1000, 1000000)) for _ in range(SAMPLE))
            }
            
            file = 'run_multiple_maximize_samples.py'
            
        case 4:
            
            variant_str, rnd_seed_str = generator_variants()
            
            args = {
                'rec_layers'  : SAMPLE_REC_LAYER,
                'scr_layers'  : SAMPLE_SCR_LAYER,
                'variant'     : variant_str,
                'random_seed' : rnd_seed_str
            }
            
            file = 'run_multiple_generator_variants.py'
            
        case _:
            
            print('Invalid option')
            
    args['name']     = NAME
    args['iter']     = str(ITER)
    args['template'] = 'T'

    copy_exec(file=file, args=args)
