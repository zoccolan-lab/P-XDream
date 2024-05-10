from typing import List
from script.utils.cmdline_args import Arg, Args


ARGS: List[Arg] = [
    
    Args.get_config_arg('optimizer_tuning.json'),
    
    # Generator
    Args.GenWeights.value,
    Args.GenVariant.value,
    
    # Subject
    Args.NetworkName.value,
    Args.RecordingLayers.value,
    
    # Scorer
    Args.ScoringLayers.value,
    Args.UnitsReduction.value,
    Args.LayerReduction.value,
    
    # Optimizer
    Args.OptimType.value,
    Args.RandomDistr.value,
    Args.RandomScale.value,
    Args.PopulationSize.value,
    Args.MutationSize.value,
    Args.MutationRate.value,
    Args.NumParents.value,
    Args.AllowClones.value,
    Args.TopK.value,
    Args.Temperature.value,
    Args.TemperatureFactor.value,
    Args.Sigma0.value,
    
    # Logger
    Args.ExperimentName.value,
    Args.ExperimentVersion.value,
    Args.OutputDirectory.value,
    
    # Globals
    Args.NumIterations.value,
    Args.DisplayPlots.value,
    Args.RandomSeed.value,
    Args.Render.value,
    
]
