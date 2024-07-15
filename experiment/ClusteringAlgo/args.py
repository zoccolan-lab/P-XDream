

from experiment.utils.args import ExperimentArgParams
from zdream.utils.parameters import ArgParams, ParamConfig


ARGS: ParamConfig = {
        
    # Clustering
    ExperimentArgParams.ClusterDir   .value: "/data/Zdream/clustering",
    ExperimentArgParams.MaxIterations.value: 50000,                                              # DS 
    ExperimentArgParams.MinElements  .value: 1,                                                  # DS
    ExperimentArgParams.UseGPU       .value: True,                                               # DS
    ExperimentArgParams.NClusters    .value: 50,                                                 # GMM & NC
    ExperimentArgParams.NComponents  .value: 500,                                                # GMM & DBSCAN
    ExperimentArgParams.Epsilon      .value: " ".join([str(e) for e in range(100, 1000, 100)]),  # GMM & DBSCAN
    ExperimentArgParams.MinSamples   .value: " ".join([str(e) for e in range(  2,   10)]),       # GMM & DBSCAN
    
    # Logger
    ArgParams.ExperimentName   .value: "cluster_algo",
    ArgParams.ExperimentVersion.value: 0,
    ArgParams.OutputDirectory  .value: "data/Zdream/output"

}