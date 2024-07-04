

from experiment.utils.args import ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig


ARGS: ParamConfig = {
        
    # Clustering
    ExperimentArgParams.ClusterDir   .value: "/data/Zdream/clustering",
    ExperimentArgParams.MaxIterations.value: 50000, # DS 
    ExperimentArgParams.MinElements  .value: 1, # DS
    ExperimentArgParams.UseGPU       .value: True, # DS
    ExperimentArgParams.NClusters    .value: 50, # GMM & NC
    ExperimentArgParams.NComponents  .value: 500, # GMM
    
    # Logger
    ArgParams.ExperimentName   .value: "cluster_algo",
    ArgParams.ExperimentVersion.value: 0,
    ArgParams.OutputDirectory  .value: "data/Zdream/output"

}