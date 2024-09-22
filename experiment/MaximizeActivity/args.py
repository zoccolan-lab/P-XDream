from experiment.utils.args import DATASET, OUT_DIR, WEIGHTS,REFERENCES, ExperimentArgParams
from pxdream.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , 
    ExperimentArgParams.GenVariant       .value : "fc7"              ,

    # Natural Images
    ExperimentArgParams.Template         .value : "T"               , 
    ExperimentArgParams.Dataset          .value : DATASET            ,
    ExperimentArgParams.Shuffle          .value : False              , 
    ExperimentArgParams.BatchSize        .value : 16                 , 

    # Subject
    ExperimentArgParams.NetworkName      .value : "alexnet"          , # resnet50
    ExperimentArgParams.RecordingLayers  .value : "21=[0]"           ,#126=[0] for rn50
    ExperimentArgParams.RobustPath      .value : '',#'/home/lorenzo/Desktop/Datafolders/imagenet_l2_3_0.pt' ,
    #, Scorer
    ExperimentArgParams.ScoringLayers    .value : "21=[0]"            ,
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,
    ExperimentArgParams.Reference   .value : REFERENCES,

    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,

    # Logger
    ArgParams          .ExperimentName   .value : "maximize_activity", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 500                ,
    ArgParams          .RandomSeed       .value : 50001              ,   #50000
    ArgParams          .Render           .value : False,

}
