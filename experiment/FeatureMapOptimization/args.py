from analysis.utils.settings import CLUSTER_DIR
from experiment.utils.args import CLUSTERING, DATASET, OUT_DIR, WEIGHTS, FEATURE_MAPS, ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig


ARGS: ParamConfig = {
    
    # Feature maps
    ExperimentArgParams.FeatureMapIdx     .value : 0,
    ExperimentArgParams.FeatureMapDir     .value : FEATURE_MAPS,
    ExperimentArgParams.FMSegmentationType.value : 'fm',
    ExperimentArgParams.FMKey             .value : '233',
    
    # Clustering 
    ExperimentArgParams.ClusterAlgo       .value : 'ds',
    ExperimentArgParams.ClusterDir        .value : CLUSTER_DIR,
    ExperimentArgParams.ClusterIdx        .value : 0,
    ExperimentArgParams.ClusterLayer      .value : 13,
    
    # Generator
    ExperimentArgParams.GenWeights        .value : WEIGHTS,
    ExperimentArgParams.GenVariant        .value : 'fc8',
    
    # Natural image dataloader
    ExperimentArgParams.Dataset           .value : DATASET,
    ExperimentArgParams.BatchSize         .value : 2,
    ExperimentArgParams.Template          .value : 'TFFFF',
    ExperimentArgParams.Shuffle           .value : False,
    
    # Subject
    ExperimentArgParams.NetworkName       .value : 'alexnet',
    
    # Scorer
    ExperimentArgParams.LayerReduction.value     : 'mean',
    ExperimentArgParams.UnitsReduction.value     : 'mean',
    
    # Optimizer
    ExperimentArgParams.PopulationSize.value     : 50,
    ExperimentArgParams.Sigma0        .value     : 1.0,
    
    # Logger
    ArgParams.OutputDirectory         .value     : OUT_DIR,
    ArgParams.ExperimentName          .value     : 'feature_map_optimization',
    ArgParams.ExperimentVersion       .value     : 0,
    
    # Experiment
    ArgParams.NumIterations           .value     : 100,
    ArgParams.RandomSeed              .value     : 123,
    ArgParams.Render                  .value     : True
    
}
