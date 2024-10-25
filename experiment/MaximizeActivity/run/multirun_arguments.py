
import random
from typing import List, Tuple

import numpy as np

from pxdream.utils.misc import copy_exec
from pxdream.utils.parameters import ArgParams
from experiment.utils.args import ExperimentArgParams
from pxdream.subject import TorchNetworkSubject


def generate_log_numbers(N, M): return list(sorted(list(set([int(a) for a in np.logspace(0, np.log10(M), N)]))))


NAME   = f'resnet50_conv53_vanilla_ref'

ITER     = 500
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

# --- REFERENCES ---
def get_rnd(
    seed=None,
    n_seeds=10,
    r_range : Tuple[Tuple[int, int], ...] | Tuple[int, int] = (1000, 1000000),
    add_parenthesis: bool = True
):
    if isinstance(r_range[0], int):
        if len(r_range) == 2:
            r_range = (r_range,)
        else: #fast parsing of tuple(tuple[int, int], ...)
            r_range = tuple((r,) for r in r_range)
    
    if seed is not None:
        random.seed(seed)
    
    unique_numbers = set()
    
    while len(unique_numbers) < n_seeds:
        idx = []
        for inner in r_range:
            if len(inner) == 1:
                start = 0
                end = inner[0]
            else:
                start, end = inner
            
            idx.append(str(random.randint(start, end)))
        if add_parenthesis == True:
            unique_numbers.add('(' + ' '.join(idx) + ')')
        else:
            unique_numbers.add(' '.join(idx))
    return list(unique_numbers)


NUM_NEURONS = 100
GLOBAL_SEED = 31415
REF_GEN_VARIANT = ['fc7']
REF_LAYERS      = [122]
REF_SEED        = get_rnd(seed=GLOBAL_SEED, n_seeds=4, add_parenthesis=False) 
NET             = 'resnet50'
ROBUST_VARIANT  = ''#'imagenet_l2_3_0.pt'
SBJ_LOADER      = 'madryLab_robust_load' if ROBUST_VARIANT else 'torch_load_pretrained'


subject = TorchNetworkSubject(
    NET,
    inp_shape=(1, 3, 224, 224),
)
layer_shape = subject.layer_shapes[REF_LAYERS[0]]
layer_shape = tuple([e-1 for e in layer_shape]) if len(layer_shape) == 2 else tuple([e-1 for e in layer_shape[1:]])
REF_NEURONS = get_rnd(seed=GLOBAL_SEED, n_seeds=NUM_NEURONS, r_range=layer_shape) #  


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

def get_args_reference() -> Tuple[str, str, str]:
    
    args = [
        (gen_var, f'{layer}=[{neuron}]', seed)
        for gen_var in REF_GEN_VARIANT
        for layer   in REF_LAYERS
        for neuron  in REF_NEURONS
        for seed    in REF_SEED
    ]
    
    gen_var_str = '#'.join(a for a, _, _ in args)
    rec_str     = '"' + '#'.join(a for _, a, _ in args) + '"'
    seed_str    = '#'.join(a for _, _, a in args)
    
    return gen_var_str, rec_str, seed_str


if __name__ == '__main__':

    print('Multiple run: ')
    print('[1] Neural scaling')
    print('[2] Layers correlation')
    print('[3] Maximize samples')
    print('[4] Generator variants')
    print('[5] Create references')
    
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
            
        case 5:
            
            gen_var_str, rec_layer_str, seed_str = get_args_reference()
            
            args = {
                str(ExperimentArgParams.GenVariant     ) : gen_var_str,
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : rec_layer_str,
                str(ExperimentArgParams.NetworkName    ) : NET,
                str(          ArgParams.RandomSeed     ) : seed_str,
                str(ExperimentArgParams.WeightLoadFunction): SBJ_LOADER
            }
            if ROBUST_VARIANT : args[str(ExperimentArgParams.CustomWeightsVariant)] = ROBUST_VARIANT
            file = 'run_multiple_references.py'
            
        case _:
            
            print('Invalid option')
            
    args[str(ArgParams          .ExperimentName)] = NAME
    args[str(ArgParams          .NumIterations )] = str(ITER)
    args[str(ExperimentArgParams.Template      )] = 'T'
    
    #fname ='/home/lorenzo/Documents/GitHub/ZXDREAM/experiment/MaximizeActivity/run/multirun_cmd.txt'

    copy_exec(file=file, args=args ) #fname = fname
