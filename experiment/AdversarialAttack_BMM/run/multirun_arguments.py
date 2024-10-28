import math
from os import path
from typing import List, Tuple

import numpy as np

from experiment.utils.args import REFERENCES, ExperimentArgParams
from pxdream.subject import TorchNetworkSubject
from pxdream.utils.io_ import load_pickle
from pxdream.utils.misc       import copy_exec
from pxdream.utils.parameters import ArgParams
from experiment.MaximizeActivity.run.multirun_arguments import get_rnd

TASK = ['adversarial', 'invariance']
NAME                = f'SnS_multiexp_271024_robustRnet50_genetic_i250'

GLOBAL_RSEED        = 50000
ITER                = 250
OPTIMIZER           = 'genetic'
TEMPLATE            = 'T'
GEN_VARIANT         = 'fc7'
NET                 = 'resnet50'
ROBUST_VARIANT  = 'imagenet_l2_3_0.pt'
SBJ_LOADER      = 'madryLab_robust_load' if ROBUST_VARIANT else 'torch_load_pretrained'
RND_SEED        = get_rnd(seed=GLOBAL_RSEED, n_seeds=10, add_parenthesis=False) 
LOW_LY              = 0
HIGH_LY             = 126
Task2Sign = {
    'invariance': f'{LOW_LY}=-1, {HIGH_LY}=1',
    'adversarial': f'{LOW_LY}=1, {HIGH_LY}=-1'
}
Task2Bound = {
    'invariance': f'{LOW_LY}=N, {HIGH_LY}=<10%',
    'adversarial': f'{LOW_LY}=N, {HIGH_LY}=>100%'
}
N_NEURONS           = 20
subject = TorchNetworkSubject(
    NET,
    inp_shape=(1, 3, 224, 224),
)
LNAME = subject.layer_names[HIGH_LY]

#select neurons and seeds
reference_file      = load_pickle(REFERENCES)
net_key = NET+'_r' if ROBUST_VARIANT else NET
refs                = reference_file['reference'][net_key][GEN_VARIANT][LNAME]
neurons_available   = list(refs.keys())
N_NEURONS = min(N_NEURONS, len(neurons_available))
neurons_idxs        = list(map(int,get_rnd(seed = GLOBAL_RSEED, n_seeds = N_NEURONS, 
                    r_range = (0,len(neurons_available)-1), add_parenthesis = False)))
NEURONS = [neurons_available[i] for i in neurons_idxs]
RSEEDS = []
for n in NEURONS:
    rseeds_available = list(refs[n].keys())
    rs_idxs = list(map(int,get_rnd(seed = GLOBAL_RSEED, n_seeds = 1, 
                    r_range = (0,len(rseeds_available)), add_parenthesis = False)))
    RSEEDS.append(rseeds_available[rs_idxs[0]])



SAMPLE = 2

def get_bmm_args():
    
    args = [
        (rs, f'{LOW_LY}=[], {HIGH_LY}={n}',
        f'G={GEN_VARIANT}, L={HIGH_LY}, N={n}, S={nrs}',
        Task2Sign[task], Task2Bound[task])
        for rs in RND_SEED
        for n,nrs in zip(NEURONS,RSEEDS)
        for task in TASK
    ]
    
    rand_seeds      = '"' + "#".join([str(rs) for rs, _, _, _, _ in args])+ '"'
    rec_score_ly    = '"' + "#".join([str(n) for _, n, _, _, _ in args])+ '"'
    ref_p           = '"' + "#".join([str(ref) for _, _, ref, _, _ in args])+ '"'
    signatures      = '"' + "#".join([str(sign) for _, _, _, sign, _ in args])+ '"'
    bounds          = '"' + "#".join([str(bound) for _, _, _, _, bound in args]) + '"'

    
    return rand_seeds, rec_score_ly, ref_p, signatures, bounds
    
    

if __name__ == '__main__':
    
    args = {}
    
    print('Multiple run: ')
    print('[1] BMM multi experiment')
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            rand_seeds, rec_score_ly, ref_p, signatures, bounds = get_bmm_args()
            
            args[str(ArgParams.RandomSeed)]                 = rand_seeds
            args[str(ExperimentArgParams.RecordingLayers)]  = rec_score_ly
            args[str(ExperimentArgParams.ScoringLayers)]    = rec_score_ly
            args[str(ExperimentArgParams.ReferenceInfo)]    = ref_p
            args[str(ExperimentArgParams.ScoringSignature)] = signatures
            args[str(ExperimentArgParams.Bounds)]           = bounds
            file = 'run_multi.py'
            
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
    args[str(          ArgParams.NumIterations )] = ITER
    args[str(          ArgParams.ExperimentName)] = NAME
    args[str(ExperimentArgParams.Template      )] = TEMPLATE
    args[str(ExperimentArgParams.GenVariant    )] = GEN_VARIANT
    args[str(ExperimentArgParams.NetworkName   )] = NET
    if ROBUST_VARIANT : args[str(ExperimentArgParams.CustomWeightsVariant)] = ROBUST_VARIANT
    args[str(ExperimentArgParams.WeightLoadFunction)] = SBJ_LOADER
    args[str(ExperimentArgParams.OptimType)] = OPTIMIZER
    
    copy_exec(file=file, args=args)
