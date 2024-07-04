from typing import List
from experiment.utils.args import ArgParam, Args


ARGS: List[ArgParam] = [
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