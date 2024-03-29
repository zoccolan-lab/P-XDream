from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from zdream.clustering.model import AffinityMatrix, PairwiseSimilarity
from zdream.clustering.algo import BaseDSClustering
from zdream.experiment import Experiment
from zdream.logger import Logger, LoguruLogger, MutedLogger
from zdream.message import Message

class DSClusteringExperiment(Experiment):
    
    EXPERIMENT_TITLE = 'DSClusteringExperiment'
    
    def __init__(
        self,
        matrix  : NDArray,
        name    : str = 'ds_clustering', 
        logger  : Logger = MutedLogger(),
        data    : Dict[str, Any] = dict()
    ) -> None:
        
        super().__init__(name, logger, data)
        
        logger.info('Computing cosine similarity...')
        self._aff_mat   = PairwiseSimilarity(matrix=matrix).cosine_similarity
        
        self._clu_algo = BaseDSClustering(
            aff_mat=self._aff_mat, 
            max_iter=data['max_iter'], 
            logger=self._logger
        )
        
    @classmethod
    def _from_config(cls, conf: Dict[str, Any]) -> 'DSClusteringExperiment':
        
        clu_conf = conf['clustering']
        log_conf = conf['logger']
        
        # --- MATRIX ---
        matrix = np.load(clu_conf['recordings'])
        
        # --- LOGGER ---
        log_conf['title'] = cls.EXPERIMENT_TITLE
        logger = LoguruLogger(conf=log_conf)
        
        # --- DATA ---
        data = {'max_iter': clu_conf['max_iter']}
        
        return DSClusteringExperiment(
            matrix=matrix,
            logger=logger,
            name=log_conf['name'],
            data=data
        )
        
    @property
    def _components(self) -> List[Tuple[str, Any]]:
        return [('AffinityMatrix', self._aff_mat)]


    def _run(self, msg: Message) -> Message:
        
        self._clu_algo.run()
        
        return msg
        
    def _finish(self, msg: Message) -> Message:
        
        msg = super()._finish(msg)
        
        # Save
        self._clu_algo.clusters.dump(
            out_fp=self.target_dir,
            logger=self._logger
        )
        
        return msg
