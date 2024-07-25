
import random
from typing import List, Tuple

import numpy as np

from zdream.utils.misc import copy_exec
from zdream.utils.parameters import ArgParams
from experiment.utils.args import ExperimentArgParams

def generate_log_numbers(N, M): return list(sorted(list(set([int(a) for a in np.logspace(0, np.log10(M), N)]))))


NAME   = f'sample_max'

ITER     = 15
SAMPLE   =  30

# --- NEURON SCALING ---

N_POINTS =  45
LAYERS = [
    (13, 9216), # Conv5 MaxPool
    (17, 4096), # FC6 Relu
    (20, 4096), # FC7 Relu
    (21, 1000)  # FC8 Maxpool
]

# --- MULTIPLE SAMPLES ---
UNITS = [573, 999]
MS_REC_SCR   = [(f"21=[{u}]", "21=[]") for u in UNITS]

# --- VARIANTS ---

# VARIANTS = ['fc8', 'fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'norm2', 'norm1'] 
VARIANTS = ['fc8', 'fc7', 'fc6'] 
VARIANT_LAYER = 21
VARIANT_NEURONS = list(range(2))

def get_rnd(): return str(random.randint(1000, 1000000))

def get_args_neuron_scaling() -> Tuple[str, str, str]:

    args = [
        (f'{layer}={neuron}r[]', f'{layer}=[]', get_rnd())
        for layer, neurons in LAYERS
        for neuron in generate_log_numbers(N_POINTS, neurons)
        for _ in range(SAMPLE)
    ]

    rec_layer_str = '#'.join(a for a, _, _ in args)
    scr_layer_str = '#'.join(a for _, a, _ in args)
    rand_seed_str = '#'.join(a for _, _, a in args)
    
    return rec_layer_str, scr_layer_str, rand_seed_str


def get_args_layers_correlation() -> Tuple[str, str, str]:

    rec_layers_str = ','.join([f'{layer}=[]' for layer in LAYERS])
    
    args = [
        ( f'{layer}={neuron}r[]', str(random.randint(1000, 1000000)))
        for layer, neu_len in LAYERS
        for neuron in generate_log_numbers(N_POINTS, neu_len)
        for _ in range(SAMPLE)]
    
    scr_layers_str = '#'.join(a for a, _ in args)
    rand_seed_str  = '#'.join(a for _, a in args)
    
    return rec_layers_str, scr_layers_str, rand_seed_str

def get_args_multi_sample() -> Tuple[str, str, str]:

    args = [
        (rec, scr, str(random.randint(1000, 1000000)))
        for rec, scr in MS_REC_SCR
        for _ in range(SAMPLE)
    ]
    
    rec_str        = '#'.join(a for a, _, _ in args)
    scr_str        = '#'.join(a for _, a, _ in args)
    rand_seed_str  = '#'.join(a for _, _, a in args)
    
    return rec_str, scr_str, rand_seed_str


def get_args_generator_variants() -> Tuple[str, str, str]:

    args = [
        (f'{VARIANT_LAYER}=[{neuron}]', f'{variant}', str(random.randint(1000, 1000000)) )
        for neuron in VARIANT_NEURONS
        for variant in VARIANTS
        for _ in range(SAMPLE)
    ]
    
    rec_str        = '#'.join(a for a, _, _ in args)
    variant_str    = '#'.join(a for _, a, _ in args)
    rand_seed_str  = '#'.join(a for _, _, a in args)
    
    return rec_str, variant_str, rand_seed_str


if __name__ == '__main__':

    print('Multiple run: ')
    print('[1] Neural scaling')
    print('[2] Layers correlation')
    print('[3] Maximize samples')
    print('[4] Generator variants')
    
    option = int(input('Choose option: '))
    
    match option:
        
        case 1:
            
            rec_layer_str, scr_layer_str, rnd_seed_str = get_args_neuron_scaling()
            
            args = {
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : scr_layer_str,
                str(          ArgParams.RandomSeed     ) : rnd_seed_str,
            }
            
            file = 'run_multiple_neuronal_scaling.py'
            
        case 2:
            
            rec_layer_str, scr_layer_str, rnd_seed_str = get_args_layers_correlation()
            
            args = {
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : scr_layer_str,
                str(          ArgParams.RandomSeed     ) : rnd_seed_str,
            }
            
            file = 'run_multiple_layer_correlation.py'
            
        case 3:
            
            rec_str, scr_str, rand_seed_str = get_args_multi_sample()
                        
            args = {
                str(ExperimentArgParams.RecordingLayers)  : rec_str,
                str(ExperimentArgParams.ScoringLayers  )  : scr_str,
                str(          ArgParams.RandomSeed     )  : rand_seed_str
            }
             
            file = 'run_multiple_maximize_samples.py'
            
        case 4:
            
            rec_layer_str, variant_str, rnd_seed_str = get_args_generator_variants()
            
            args = {
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : f'{VARIANT_LAYER}=[]',
                str(ExperimentArgParams.GenVariant     ) : variant_str,
                str(          ArgParams.RandomSeed     ) : rnd_seed_str
            }
            
            file = 'run_multiple_generator_variants.py'
            
        case _:
            
            print('Invalid option')
            
    args[ArgParams          .ExperimentName] = NAME
    args[ArgParams          .NumIterations ] = str(ITER)
    args[ExperimentArgParams.Template      ] = 'T'

    copy_exec(file=file, args=args)
