from experiment.utils.args import DATASET, OUT_DIR, WEIGHTS, ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig = {
    
    # Subejct
    ExperimentArgParams.NetworkName.value     : "alexnet",
    ExperimentArgParams.RecordingLayers.value : "21=[]",
    
    # Dataset
    ExperimentArgParams.Dataset.value       : DATASET,
    ExperimentArgParams.ImageIds.value      : "",
    ExperimentArgParams.LogCheckpoint.value : 1,
    
    # Logger
    ArgParams.ExperimentName.value    : "neural_recording",
    ArgParams.ExperimentVersion.value : 0,
    ArgParams.OutputDirectory.value   : OUT_DIR,
}