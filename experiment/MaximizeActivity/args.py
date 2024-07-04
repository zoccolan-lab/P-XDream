from experiment.utils.args import DATASET, OUT_DIR, WEIGHTS, ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , 
    ExperimentArgParams.GenVariant       .value : "fc8"              ,

    # Natural Images
    ExperimentArgParams.Template         .value : "TF"               , 
    ExperimentArgParams.Dataset          .value : DATASET            ,
    ExperimentArgParams.Shuffle          .value : False              , 
    ExperimentArgParams.BatchSize        .value : 16                 , 

    # Subject
    ExperimentArgParams.NetworkName      .value : "alexnet"          , 
    ExperimentArgParams.RecordingLayers  .value : "21=[0:20]"        ,

    #, Scorer
    ExperimentArgParams.ScoringLayers    .value : "21=[]"            ,
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,

    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.RandomDistr      .value : "normal"           ,
    ExperimentArgParams.RandomScale      .value : 1.0                ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,

    # Logger
    ArgParams          .ExperimentName   .value : "maximize_activity", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 150                ,
    ArgParams          .RandomSeed       .value : 50000              ,
    ArgParams          .Render           .value : True               ,

}
