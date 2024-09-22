

from experiment.utils.args import ExperimentArgParams
from pxdream.utils.parameters import ArgParams, ParamConfig


ARGS: ParamConfig = {
        
    # Clustering
    ExperimentArgParams.ClusterDir      .value: "/data/Zdream/clustering",
    ExperimentArgParams.MaxIterations   .value: 50000,                                              # DS 
    ExperimentArgParams.MinElements     .value: 1,                                                  # DS
    ExperimentArgParams.UseGPU          .value: True,                                               # DS
    ExperimentArgParams.NClusters       .value: 50,                                                 # GMM & NC
    ExperimentArgParams.NComponents     .value: 500,                                                # GMM & DBSCAN
    ExperimentArgParams.Epsilon         .value: " ".join([str(e) for e in range(100, 1000, 100)]),  # DBSCAN
    ExperimentArgParams.MinSamples      .value: " ".join([str(e) for e in range(  2,   10)]),       # DBSCAN
    ExperimentArgParams.DimReductionType.value: " ".join([ "pca", "tsne"]),                         # GMM & DBSCAN
    ExperimentArgParams.NComponents     .value: " ".join([  '50',  '100']),                         # GMM & DBSCAN
    ExperimentArgParams.TSNEPerplexity  .value: " ".join([   '3',   '30']),                         # GMM & DBSCAN
    ExperimentArgParams.TSNEIterations  .value: " ".join(['1000', '4000']),                         # GMM & DBSCAN
    
    # Logger
    ArgParams.ExperimentName   .value: "cluster_algo",
    ArgParams.ExperimentVersion.value: 0,
    ArgParams.OutputDirectory  .value: "data/Zdream/output"

}