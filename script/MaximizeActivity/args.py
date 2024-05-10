from typing import List
from script.utils.cmdline_args import Arg, Args

ARGS: List[Arg] = [
    
    Args.get_config_arg(conf_file="maximize_activity.json"),

    # Natural image dataloader
    Args.GenWeights.value, 
    Args.GenVariant.value,

    # Natural Images
    Args.Template.value, 
    Args.Dataset.value, 
    Args.Shuffle.value, 
    Args.BatchSize.value, 

    # Subject
    Args.NetworkName.value, 
    Args.RecordingLayers.value,

    #, Scorer
    Args.ScoringLayers.value,
    Args.UnitsReduction.value,
    Args.LayerReduction.value,

    # Optimizer
    Args.PopulationSize.value,
    Args.RandomDistr.value,
    Args.RandomScale.value,
    Args.Sigma0.value,

    # Logger
    Args.ExperimentName.value, 
    Args.ExperimentVersion.value, 
    Args.OutputDirectory.value, 

    # Globals
    Args.NumIterations.value,
    Args.DisplayPlots.value,
    Args.RandomSeed.value,
    Args.Render.value
]
