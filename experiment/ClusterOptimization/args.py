from experiment.utils.args import CLUSTERING, DATASET, OUT_DIR, WEIGHTS, ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig


ARGS: ParamConfig = {
    
    # Clustering 
    ExperimentArgParams.ClusterAlgo   .value : 'ds',
    ExperimentArgParams.ClusterIdx    .value : 0,
    ExperimentArgParams.WeightedScore .value : False,
    ExperimentArgParams.ScoringType   .value : 'subset',
    ExperimentArgParams.OptimUnits    .value : '1',
    ExperimentArgParams.ClusterLayer  .value : 21,
    
    # Generator
    ExperimentArgParams.GenWeights    .value : WEIGHTS,
    ExperimentArgParams.GenVariant    .value : 'fc8',
    
    # Natural image dataloader
    ExperimentArgParams.Dataset       .value : DATASET,
    ExperimentArgParams.BatchSize     .value : 2,
    ExperimentArgParams.Template      .value : 'TFFFF',
    ExperimentArgParams.Shuffle       .value : False,
    
    # Subject
    ExperimentArgParams.NetworkName   .value : 'alexnet',
    
    # Scorer
    ExperimentArgParams.LayerReduction.value : 'mean',
    
    # Optimizer
    ExperimentArgParams.PopulationSize.value : 50,
    ExperimentArgParams.Sigma0        .value : 1.0,
    
    # Logger
    ArgParams.OutputDirectory         .value : OUT_DIR,
    ArgParams.ExperimentName          .value : 'cluster_optimization',
    ArgParams.ExperimentVersion       .value : 0,
    
    # Experiment
    ArgParams.NumIterations           .value : 100,
    ArgParams.RandomSeed              .value : 123,
    ArgParams.Render                  .value : True
    
}
    