from typing import List
from experiment.utils.cmdline_args import Arg, Args


ARGS: List[Arg] = [
    Args.get_config_arg(conf_file='neural_recording.json'),
    
    # Subejct
    Args.NetworkName.value,
    Args.RecordingLayers.value,
    
    # Dataset
    Args.Dataset.value,
    Args.ImageIds.value,
    Args.LogCheckpoint.value,
    
    # Logger
    Args.ExperimentName.value,
    Args.ExperimentVersion.value,
    Args.OutputDirectory.value,
]