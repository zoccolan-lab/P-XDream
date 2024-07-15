from os import path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from analysis.utils.settings import FILE_NAMES
from experiment.ClusteringAlgo.plotting import plot_cluster_extraction_trend, plot_cluster_ranks
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
        recording_fp = path.join(PARAM_clu_dir, FILE_NAMES['recordings'])
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
        
        plot_cluster_ranks(
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
            n_components=data['n_comp'],
            logger=logger
        )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'GMMClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str (conf[ExperimentArgParams.ClusterDir   .value])
        PARAM_n_clusters   = int (conf[ExperimentArgParams.NClusters    .value])
        PARAM_n_components = int (conf[ExperimentArgParams.NComponents  .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        # --- MATRIX ---
        recording_fp = path.join(PARAM_clu_dir, FILE_NAMES['recordings'])
        matrix = np.load(recording_fp)
        
        # --- LOGGER ---
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        data = {
            'n_clu' :   PARAM_n_clusters,
            'n_comp':   PARAM_n_components
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
        recording_fp = path.join(PARAM_clu_dir, FILE_NAMES['recordings'])
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
        eps          = data['eps']
        min_samples  = data['min_samples']
        n_components = data['n_components']
        
        # Check if to apply grid search
        eps_type = type(eps)
        msp_type = type(min_samples)
        
        if any([eps_type is list, msp_type is list]):
            
            if eps_type is not list: eps         = [eps]         # type: ignore
            if msp_type is not list: min_samples = [min_samples] # type: ignore
            
            self._dbscan_algo = DBSCANClusteringAlgorithm.grid_search(
                data=matrix,
                eps=eps,                   # type: ignore
                min_samples=min_samples,   # type: ignore
                n_components=n_components,
                logger=logger
            )
        
        else:
            
            self._dbscan_algo = DBSCANClusteringAlgorithm(
                data=matrix, 
                eps=eps,                  # type: ignore
                min_samples=min_samples,  # type: ignore
                n_components=n_components, 
                logger=logger
            )
        
    @classmethod
    def _from_config(cls, conf: ParamConfig) -> 'DBSCANClusteringExperiment':
        
        # Clustering
        PARAM_clu_dir      = str(conf[ExperimentArgParams.ClusterDir .value])
        PARAM_n_components = int(conf[ExperimentArgParams.NComponents.value])
        PARAM_eps          = str(conf[ExperimentArgParams.Epsilon    .value])
        PARAM_msp          = str(conf[ExperimentArgParams.MinSamples .value])
        
        # Logger
        PARAM_exp_name     = str(conf[ArgParams.ExperimentName   .value])
        
        # --- MATRIX ---
        
        recording_fp = path.join(PARAM_clu_dir, FILE_NAMES['recordings'])
        matrix       = np.load(recording_fp)
        
        # --- LOGGER ---
        
        conf[ArgParams.ExperimentTitle.value] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- DATA ---
        eps = [float(e) for e in PARAM_eps.split(' ')]
        msp = [int  (m) for m in PARAM_msp.split(' ')]
        
        if len(eps) == 1: eps = eps[0]
        if len(msp) == 1: msp = msp[0]
        
        data = {
            'eps'         : eps,
            'min_samples' : msp,
            'n_components': PARAM_n_components
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
