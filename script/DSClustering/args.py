from script.utils.cmdline_args import Args


ARGS = [
    Args.get_config_arg(conf_file='ds_clustering.json'),
    
    # Clustering
    Args.Recordings.value,
    Args.MaxIterations.value,
    Args.MinElements.value,
    Args.UseGPU.value,
    
    # Logger
    Args.ExperimentName.value,
    Args.ExperimentVersion.value,
    Args.OutputDirectory.value,
]