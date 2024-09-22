from typing import List

from experiment.utils.args import OUT_DIR, WEIGHTS, ExperimentArgParams
from pxdream.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig =  {
    
    # Generator
    ExperimentArgParams.GenWeights       .value: WEIGHTS,
    ExperimentArgParams.GenVariant       .value: 'fc8',
    ExperimentArgParams.NetworkName      .value: 'alexnet',
    
    # Subject
    ExperimentArgParams.RecordingLayers  .value: '21=[0:20]',
    
    # Scorer
    ExperimentArgParams.ScoringLayers    .value: '[]',
    ExperimentArgParams.UnitsReduction   .value: 'mean',
    ExperimentArgParams.LayerReduction   .value: 'mean',
    
    # Optimizer
    ExperimentArgParams.OptimType        .value: 'genetic',
    ExperimentArgParams.RandomDistr      .value: 'normal',
    ExperimentArgParams.RandomScale      .value: 1.0,
    ExperimentArgParams.PopulationSize   .value: 50,
    ExperimentArgParams.MutationSize     .value: 0.5,
    ExperimentArgParams.MutationRate     .value: 0.5,
    ExperimentArgParams.NumParents       .value: 4,
    ExperimentArgParams.AllowClones      .value: True,
    ExperimentArgParams.TopK             .value: 5,
    ExperimentArgParams.Temperature      .value: 1.0,
    ExperimentArgParams.TemperatureFactor.value: 0.9999,
    ExperimentArgParams.Sigma0           .value: 1.0,
    
    # Logger
    ArgParams          .ExperimentName   .value: "optimization_tuning",
    ArgParams          .ExperimentVersion.value: 0,
    ArgParams          .OutputDirectory  .value: OUT_DIR,    
    
    # Globals
    ArgParams          .NumIterations    .value: 150,
    ArgParams          .DisplayPlots     .value: True,
    ArgParams          .RandomSeed       .value: 123,
    ArgParams          .Render           .value: True,

}