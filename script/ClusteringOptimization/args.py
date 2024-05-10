from typing import List

from script.utils.cmdline_args import Args, Arg

ARGS: List[Arg] = [
    
    Args.get_config_arg(conf_file='clustering_optimization.json'),
    
    # Clustering
    Args.ClusterFile.value,
    Args.ClusterIdx.value,
    Args.WeightedScore.value,
    Args.ClusterLayer.value,
    Args.ScoringType.value,
    Args.OptimUnits.value,
    
    # Generator
    Args.GenWeights.value,
    Args.GenVariant.value,
    
    # Natural Images
    Args.Template.value, 
    Args.Dataset.value, 
    Args.Shuffle.value, 
    Args.BatchSize.value, 

    # Subject
    Args.NetworkName.value,
    
    # Scorer
    Args.LayerReduction.value,
    
    # Optimizer
    Args.RandomDistr.value,
    Args.RandomScale.value,
    Args.PopulationSize.value,
    Args.Sigma0.value,
    
    # Logger
    Args.ExperimentName.value,
    Args.ExperimentVersion.value,
    Args.OutputDirectory.value,
    
    # Globals
    Args.NumIterations.value,
    Args.RandomSeed.value,
    Args.Render.value,
    
]