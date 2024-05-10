from argparse import ArgumentParser
from os import path
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
from numpy.typing import NDArray

from script.DSClustering.plotting import plot_cluster_extraction_trend, plot_cluster_ranks
from script.utils.cmdline_args import Args
from script.utils.utils import make_dir
from zdream.clustering.model import AffinityMatrix, PairwiseSimilarity
from zdream.clustering.algo import BaseDSClustering, BaseDSClusteringGPU
from zdream.experiment import Experiment
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger
from zdream.utils.message import Message

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
    def _from_config(cls, conf: Dict[str, Any]) -> 'DSClusteringExperiment':
        
        clu_conf = conf['clustering']
        log_conf = conf['logger']
        
        # --- MATRIX ---
        matrix = np.load(clu_conf[str(Args.Recordings)])
        
        # --- CLUSTERING ---
        print(clu_conf)
        DSAlgo = BaseDSClusteringGPU if clu_conf[str(Args.UseGPU)] else BaseDSClustering
        
        # --- LOGGER ---
        log_conf[str(Args.ExperimentTitle)] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(path=log_conf)
        
        # --- DATA ---
        data = {
            'min_elements': clu_conf[str(Args.MinElements)],
            'max_iter':     clu_conf[str(Args.MaxIterations)]
        }
        
        return DSClusteringExperiment(
            matrix=matrix,
            DSAlgo=DSAlgo,
            logger=logger,
            name=log_conf[str(Args.ExperimentName)],
            data=data
        )
    
    @property
    def _components(self) -> List[Tuple[str, Any]]: return [('AffinityMatrix', self._aff_mat)]


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
