from experiment.utils.cmdline_args import Args


ARGS = [
    Args.get_config_arg(conf_file='clustering_algo.json'),
    
    # Clustering
    Args.ClusterDir   .value,
    Args.MaxIterations.value, # DS 
    Args.MinElements  .value, # DS
    Args.UseGPU       .value, # DS
    Args.NClusters    .value, # GMM & NC
    Args.NComponents  .value, # GMM
    
    # Logger
    Args.ExperimentName   .value,
    Args.ExperimentVersion.value,
    Args.OutputDirectory  .value,
]