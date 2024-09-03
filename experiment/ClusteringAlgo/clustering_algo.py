from os import path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from experiment.ClusteringAlgo.plotting import plot_cluster_extraction_trend, plot_cluster_extraction_ranks
from experiment.utils.args import ExperimentArgParams
from experiment.utils.misc import make_dir
from zdream.clustering.model import PairwiseSimilarity
from zdream.clustering.algo import BaseDSClustering, BaseDSClusteringGPU, DBSCANClusteringAlgorithm, GaussianMixtureModelsClusteringAlgorithm, NormalizedCutClusteringAlgorithm
from zdream.experiment import Experiment
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger
from zdream.utils.message import Message
from zdream.utils.parameters import ArgParams, ParamConfig

class DSClusteringExperiment(Experiment):
    
    EXPERIMENT_TITLE = 'DSClustering'
    
    def __init__(
        self,
        matrix : NDArray,
        DSAlgo : Type[BaseDSClustering],
        name   : str = 'ds_clustering', 
        logger : Logger = SilentLogger(),
        data   : Dict[str, Any] = dict()
    ) -> None:
        
        super().__init__(name, logger, data)
        
        # Compute cosine similarity
        logger.info('Computing cosine similarity...')
        self._aff_mat = PairwiseSimilarity.cosine_similarity(matrix=matrix)
        
        # Save affinity matrix
        aff_mat_fp = path.join(self.dir, 'affinity_matrix.npy')
        logger.info(mess=f'Saving numpy matrix to {aff_mat_fp}')
        np.save(file=aff_mat_fp, arr=self._aff_mat.A)
        
        self._clu_algo = DSAlgo(
            aff_mat=self._aff_mat, 
            min_elements=data['min_elements'],
            max_iter=data['max_iter'], 
            logger=self._logger
        )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'DSClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str (conf[ExperimentArgParams.ClusterDir   .value])
        PARAM_max_iter     = int (conf[ExperimentArgParams.MaxIterations.value]) 
        PARAM_min_el       = int (conf[ExperimentArgParams.MinElements  .value])
        PARAM_use_gpu      = bool(conf[ExperimentArgParams.UseGPU       .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        
        # --- MATRIX ---
        recording_fp = path.join(PARAM_clu_dir, 'recordings.npy')
        matrix = np.load(recording_fp)
        
        # --- CLUSTERING ---
        DSAlgo = BaseDSClusteringGPU if PARAM_use_gpu else BaseDSClustering
        
        # --- LOGGER ---
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        data = {
            'min_elements': PARAM_min_el,
            'max_iter':     PARAM_max_iter
        }
        
        return DSClusteringExperiment(
            matrix=matrix,
            DSAlgo=DSAlgo,
            logger=logger,
            name=PARAM_exp_name,
            data=data
        )
    
    @property
    def _components(self) -> List[Tuple[str, Any]]: return [('DSClustering', self._clu_algo)]

    def _run(self, msg: Message) -> Message:
        
        self._clu_algo.run()
        
        return msg
        
    def _finish(self, msg: Message) -> Message:
        
        msg = super()._finish(msg)
        
        # Extract clusters
        clusters = self._clu_algo.clusters
        
        # Save
        clusters.dump(
            out_fp=self.dir,
            logger=self._logger
        )
        
        # Plots
        plot_dir = make_dir(path=path.join(self.dir, 'plots'), logger=self._logger)
        
        self._logger.formatting = lambda x: f'> {x}'

        plot_cluster_extraction_trend(
            clusters=clusters,
            out_dir=plot_dir,
            logger=self._logger
        )
        
        plot_cluster_extraction_ranks(
            clusters=clusters,
            out_dir=plot_dir,
            logger=self._logger
        )
        
        self._logger.reset_formatting()
        
        return msg

class GMMClusteringExperiment(Experiment):
    
    EXPERIMENT_TITLE = 'GMMClustering'
    
    def __init__(
        self,
        matrix : NDArray,
        name   : str = 'gmm_clustering', 
        logger : Logger = SilentLogger(),
        data   : Dict[str, Any] = dict()
    ) -> None:
        
        super().__init__(name, logger, data)
        
        self._gmm_algo = GaussianMixtureModelsClusteringAlgorithm(
            data=matrix,
            n_clusters=data['n_clu'],
            dim_reduction=data['dim_reduction'],
            logger=logger
        )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'GMMClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str(conf[ExperimentArgParams.ClusterDir      .value])
        PARAM_n_clusters   = int(conf[ExperimentArgParams.NClusters       .value])
        PARAM_n_components = str(conf[ExperimentArgParams.NComponents     .value])
        PARAM_dim_red_type = str(conf[ExperimentArgParams.DimReductionType.value])
        PARAM_tsne_perp    = str(conf[ExperimentArgParams.TSNEPerplexity  .value])
        PARAM_tsne_iter    = str(conf[ExperimentArgParams.TSNEIterations  .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        # --- MATRIX ---
        recording_fp = path.join(PARAM_clu_dir, 'recordings.npy')
        matrix = np.load(recording_fp)
        
        # --- LOGGER ---
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        
        drt   = [str(d) for d in PARAM_dim_red_type.split(' ')]
        nco   = [int(n) for n in PARAM_n_components.split(' ')]
        prp   = [int(p) for p in PARAM_tsne_perp.split(' ')]
        niter = [int(i) for i in PARAM_tsne_iter.split(' ')]
        
        if len(drt) == 1: drt = drt[0] 
        else: raise ValueError(f'Only one dimensionality reduction type is allowed for parameter `dim_reduction`, but got multiple: {drt}')
        
        if len(nco) == 1: nco = nco[0] 
        else: raise ValueError(f'Only one number of components is allowed for parameter `n_components`, but got multiple: {nco}')
        
        if len(prp) == 1: prp = prp[0]
        else: raise ValueError(f'Only one perplexity is allowed for parameter `tsne_perplexity`, but got multiple: {prp}')
        
        if len(niter) == 1: niter = niter[0]
        else: raise ValueError(f'Only one number of iterations is allowed for parameter `tsne_iterations`, but got multiple: {niter}')
            
        data = {
            'n_clu': PARAM_n_clusters,
            'dim_reduction': {
                'type'        : drt,
                'n_components': nco,
                'perplexity'  : prp,
                'n_iter'      : niter
            }
        }
        
        return GMMClusteringExperiment(
            matrix=matrix,
            logger=logger,
            name=PARAM_exp_name,
            data=data
        )
    
    @property
    def _components(self) -> List[Tuple[str, Any]]: return [('GMMClustering', self._gmm_algo)]

    def _run(self, msg: Message) -> Message:
        
        self._gmm_algo.run()
        
        return msg
        
    def _finish(self, msg: Message) -> Message:
        
        msg = super()._finish(msg)
        
        # Extract clusters
        clusters = self._gmm_algo.clusters
        
        # Save
        clusters.dump(
            out_fp=self.dir,
            logger=self._logger
        )
        
        return msg

class NCClusteringExperiment(Experiment):
    
    EXPERIMENT_TITLE = 'NCClustering'
    
    def __init__(
        self,
        matrix  : NDArray,
        name    : str = 'nc_clustering', 
        logger  : Logger = SilentLogger(),
        data    : Dict[str, Any] = dict()
    ) -> None:
        
        super().__init__(name, logger, data)
        
        # Compute cosine similarity
        logger.info('Computing cosine similarity...')
        self._aff_mat = PairwiseSimilarity.cosine_similarity(matrix=matrix)
        
        # Save affinity matrix
        aff_mat_fp = path.join(self.dir, 'affinity_matrix.npy')
        logger.info(mess=f'Saving numpy matrix to {aff_mat_fp}')
        np.save(file=aff_mat_fp, arr=self._aff_mat.A)
        
        self._nc_algo = NormalizedCutClusteringAlgorithm(
            aff_mat=self._aff_mat,
            n_clusters=data['n_clu'],
            logger=logger
        )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'NCClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str (conf[ExperimentArgParams.ClusterDir   .value])
        PARAM_n_clusters   = int (conf[ExperimentArgParams.NClusters    .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        # --- MATRIX ---
        recording_fp = path.join(PARAM_clu_dir, 'recordings.npy')
        matrix = np.load(recording_fp)
        
        # --- LOGGER ---
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        data = {
            'n_clu' :   PARAM_n_clusters
        }
        
        return NCClusteringExperiment(
            matrix=matrix,
            logger=logger,
            name=PARAM_exp_name,
            data=data
        )
    
    @property
    def _components(self) -> List[Tuple[str, Any]]: return [('NCClustering', self._nc_algo)]

    def _run(self, msg: Message) -> Message:
        
        self._nc_algo.run()
        
        return msg
        
    def _finish(self, msg: Message) -> Message:
        
        msg = super()._finish(msg)
        
        # Extract clusters
        clusters = self._nc_algo.clusters
        
        # Save
        clusters.dump(
            out_fp=self.dir,
            logger=self._logger
        )
        
        return msg


class DBSCANClusteringExperiment(Experiment):
    
    EXPERIMENT_TITLE = 'DBSCANClustering'
    
    def __init__(
        self,
        matrix      : NDArray,
        name    : str            = 'dbscan_clustering', 
        logger  : Logger         = SilentLogger(),
        data    : Dict[str, Any] = dict()
    ) -> None:
        
        super().__init__(name, logger, data)
        
        # Extract data
        eps           = data['eps']
        min_samples   = data['min_samples']
        dim_reduction = data['dim_reduction']
        
        # Check if to apply grid search
        eps_type = type(eps)
        msp_type = type(min_samples)
        drd_type = type(dim_reduction)
        
        if any([eps_type is list, msp_type is list, drd_type is list]):
            
            if eps_type is not list: eps           = [eps]         # type: ignore
            if msp_type is not list: min_samples   = [min_samples] # type: ignore
            if drd_type is not list: dim_reduction = [dim_reduction]
            
            self._dbscan_algo = DBSCANClusteringAlgorithm.grid_search(
                data=matrix,
                eps=eps,                   # type: ignore
                min_samples=min_samples,   # type: ignore
                dim_reduction=dim_reduction,
                logger=logger
            )
        
        else:
            
            self._dbscan_algo = DBSCANClusteringAlgorithm(
                data=matrix, 
                eps=eps,                  # type: ignore
                min_samples=min_samples,  # type: ignore
                dim_reduction=dim_reduction, 
                logger=logger
            )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'DBSCANClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str(conf[ExperimentArgParams.ClusterDir      .value])
        PARAM_eps          = str(conf[ExperimentArgParams.Epsilon         .value])
        PARAM_msp          = str(conf[ExperimentArgParams.MinSamples      .value])
        PARAM_dim_red_type = str(conf[ExperimentArgParams.DimReductionType.value])
        PARAM_n_components = str(conf[ExperimentArgParams.NComponents     .value])
        PARAM_tsne_perp    = str(conf[ExperimentArgParams.TSNEPerplexity  .value])
        PARAM_tsne_iter    = str(conf[ExperimentArgParams.TSNEIterations  .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        # --- MATRIX ---
        
        recording_fp = path.join(PARAM_clu_dir, 'recordings.npy')
        matrix       = np.load(recording_fp)
        
        # --- LOGGER ---
        
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        
        eps = [float(e) for e in PARAM_eps.split(' ')]
        msp = [int  (m) for m in PARAM_msp.split(' ')]
        drt = [str  (d) for d in PARAM_dim_red_type.split(' ')]
        nco = [int  (n) for n in PARAM_n_components.split(' ')]
        prp = [int  (p) for p in PARAM_tsne_perp.split(' ')]
        itr = [int  (i) for i in PARAM_tsne_iter.split(' ')]
        
        dre = [
            {
                'type'      : drt_,
                'n_components': nco_,
                'perplexity'  : prp_,
                'iterations'  : itr_
            }
            for drt_, nco_, prp_, itr_ in zip(drt, nco, prp, itr)
        ]
        
        if len(eps) == 1: eps = eps[0]
        if len(msp) == 1: msp = msp[0]
        if len(dre) == 1: dre = dre[0]
        
        data = {
            'eps'           : eps,
            'min_samples'   : msp,
            'dim_reduction' : dre,
            'n_components'  : PARAM_n_components
        }
        
        return DBSCANClusteringExperiment(
            matrix=matrix,
            logger=logger,
            name=PARAM_exp_name,
            data=data
        )
    
    @property
    def _components(self) -> List[Tuple[str, Any]]: return [('NCClustering', self._dbscan_algo)]

    def _run(self, msg: Message) -> Message:
        
        self._dbscan_algo.run()
        
        return msg
        
    def _finish(self, msg: Message) -> Message:
        
        msg = super()._finish(msg)
        
        # Extract clusters
        clusters = self._dbscan_algo.clusters
        
        # Save
        clusters.dump(out_fp=self.dir, logger=self._logger)
        
        return msg
