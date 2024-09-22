from experiment.utils.args import DATASET, OUT_DIR, WEIGHTS, REFERENCES, ExperimentArgParams
from pxdream.utils.parameters import ArgParams, ParamConfig


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
    # ExperimentArgParams.NetworkName      .value : "resnet50"                                            , # resnet50
    ExperimentArgParams.NetworkName      .value : "alexnet"                                             , # resnet50
    #ExperimentArgParams.RecordingLayers  .value : "0=[], 126=[0]"                                       , # 126 resnet50
    ExperimentArgParams.RecordingLayers  .value : "0=[], 21=[0]"                                       , # 126 resnet50
    ExperimentArgParams.RobustPath       .value : '/home/lorenzo/Desktop/Datafolders/imagenet_l2_3_0.pt', #'/home/lorenzo/Desktop/Datafolders/imagenet_l2_3_0.pt' ,

    # Scorer
    # ExperimentArgParams.ScoringLayers    .value : "0=[],126=[0]"           ,
    # ExperimentArgParams.ScoringSignature .value : "0=-1, 126=1"       ,
    # ExperimentArgParams.Bounds           .value : "0=N<10, 126=<10%"       ,
    ExperimentArgParams.ScoringLayers    .value : "0=[],21=[]"           ,
    ExperimentArgParams.ScoringSignature .value : "0=-1, 21=1"       ,
    ExperimentArgParams.Bounds           .value : "0=N<10, 21=<10%"       ,
    ExperimentArgParams.Distance         .value : "euclidean"        ,
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,
    ExperimentArgParams.Reference        .value : REFERENCES ,
    ExperimentArgParams.ReferenceInfo    .value : "L=fc8, N=0, S=123"  ,
    
    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,

    # Logger
    ArgParams          .ExperimentName   .value : "adversarial_attack", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 250              ,
    ArgParams          .RandomSeed       .value : 50000              ,
    ArgParams          .Render           .value : False,

}