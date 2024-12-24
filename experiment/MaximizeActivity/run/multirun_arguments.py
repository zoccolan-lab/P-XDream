
from functools import reduce
import operator
import random
from typing import List, Set, Tuple

import numpy as np

from pxdream.utils.io_ import load_pickle
from pxdream.utils.misc import copy_exec
from pxdream.utils.parameters import ArgParams
from experiment.utils.args import REFERENCES, ExperimentArgParams
from pxdream.subject import TorchNetworkSubject


def generate_log_numbers(N, M): return list(sorted(list(set([int(a) for a in np.logspace(0, np.log10(M), N)]))))


NAME   = f'101224_refs_Robust_ConvNeXtB_100_readoutL300'

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
    r_range: Tuple[Tuple[int, int], ...] | Tuple[int, int] = (1, 10),
    add_parenthesis: bool = True,
    avoid_numbers: Set[int] | Tuple[Set[int],...] = None
):
    if isinstance(r_range[0], int):
        if len(r_range) == 2:
            r_range = (r_range,)
        elif isinstance(r_range[0], int):
            r_range = tuple((0,r) for r in r_range)  
        else:  # fast parsing of tuple(tuple[int, int], ...)
            r_range = tuple((r,) for r in r_range)

    if seed is not None:
        random.seed(seed)

    if avoid_numbers is None:
        avoid_numbers = tuple(set() for _ in range(len(r_range)))

    if not isinstance(avoid_numbers, tuple): avoid_numbers =(avoid_numbers,)
    # Calculate the total number of possible unique numbers
    total_possible_numbers = reduce(operator.mul,[end - start + 1 - len(avoid_numbers[i]) 
                                                  for i,(start, end) in enumerate(r_range)])
    # Security check
    if n_seeds > total_possible_numbers:
        raise ValueError("Requested more unique numbers than possible to sample given the range and avoid_numbers.")


    unique_numbers = set()

    while len(unique_numbers) < n_seeds:
        idx = []
        for i,inner in enumerate(r_range):
            start, end = inner
            an = avoid_numbers[i]

            # Generate a random number that is not in avoid_numbers
            while True:
                rand_num = random.randint(start, end)
                if rand_num not in an:
                    break

            idx.append(str(rand_num))

        if add_parenthesis:
            unique_numbers.add('(' + ' '.join(idx) + ')')
        else:
            unique_numbers.add(' '.join(idx))

    return list(unique_numbers)

NET             = 'convnext_base'
ROBUST_VARIANT  = 'Liu2023Comprehensive_ConvNeXt-B'
REF_GEN_VARIANT = ['fc7']
REF_LAYERS      = [156]

subject = TorchNetworkSubject(
    NET,
    inp_shape=(1, 3, 224, 224),
)
LNAME = subject.layer_names[REF_LAYERS[0]]
layer_shape = subject.layer_shapes[REF_LAYERS[0]]

reference_file      = load_pickle(REFERENCES)
net_key = NET+'_r' if ROBUST_VARIANT else NET
try:
    refs                = reference_file['reference'][net_key][REF_GEN_VARIANT[0]][LNAME]
    neurons_present = set([int(key.strip('[]')) for key in refs.keys()])
except:
    neurons_present = None

NUM_NEURONS = 100
GLOBAL_SEED = 31415
REF_SEED        = get_rnd(seed=GLOBAL_SEED, n_seeds=4, add_parenthesis=False) 
if ROBUST_VARIANT:
    if ROBUST_VARIANT == 'imagenet_l2_3_0.pt':
        SBJ_LOADER = 'madryLab_robust_load'
    else:
        SBJ_LOADER = 'robustBench_load'
else:
    SBJ_LOADER = 'torch_load_pretrained'
print('layer shape',layer_shape)
layer_shape = tuple([e-1 for e in layer_shape]) if len(layer_shape) == 2 else tuple([e-1 for e in layer_shape[1:]])
REF_NEURONS = get_rnd(seed=GLOBAL_SEED, n_seeds=NUM_NEURONS, r_range=layer_shape, avoid_numbers = neurons_present) #  


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
