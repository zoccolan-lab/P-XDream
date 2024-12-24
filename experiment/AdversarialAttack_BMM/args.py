from experiment.utils.args import DATASET, OUT_DIR, WEIGHTS, REFERENCES, CUSTOM_WEIGHTS, ExperimentArgParams
from pxdream.utils.parameters import ArgParams, ParamConfig
import os

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , 
    ExperimentArgParams.GenVariant       .value : "fc7"              ,

    # Natural Images
    ExperimentArgParams.Template         .value : "T"                , 
    ExperimentArgParams.Dataset          .value : DATASET            ,
    ExperimentArgParams.Shuffle          .value : False              , 
    ExperimentArgParams.BatchSize        .value : 16                 , 

    # Subject
    ExperimentArgParams.NetworkName             .value : 'alexnet',        # resnet50
    ExperimentArgParams.RecordingLayers         .value : "0=[], 21=[0]"  , # 126 resnet50
    ExperimentArgParams.CustomWeightsPath       .value : CUSTOM_WEIGHTS, 
    ExperimentArgParams.CustomWeightsVariant    .value : '', # 'imagenet_l2_3_0.pt'
    ExperimentArgParams.WeightLoadFunction.value       : 'torch_load',
    
    # Scorer
    ExperimentArgParams.ScoringLayers    .value : "0=[],21=[]"           ,
    ExperimentArgParams.ScoringSignature .value : "0=-1, 21=1"       ,
    ExperimentArgParams.Bounds           .value : "0=N<10, 21=<10%"       ,
    ExperimentArgParams.Distance         .value : "euclidean"        ,
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,
    ExperimentArgParams.Reference        .value : REFERENCES ,
    ExperimentArgParams.ReferenceInfo    .value : "G=fc7, L=21, N=[348], S=482726"  ,
    
    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,
    ExperimentArgParams.OptimType        .value : 'hybrid'            ,
    ExperimentArgParams.Noise_strength   .value : 0.01              ,#0.01

    # Logger
    ArgParams          .ExperimentName   .value : "adversarial_attack", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 15              ,
    ArgParams          .RandomSeed       .value : 50000              ,
    ArgParams          .Render           .value : False,

}