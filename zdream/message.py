

# --- MESSAGE ---

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from zdream.utils.misc import SEM
from zdream.utils.model import Codes, Mask, RFBox, RecordingUnit, ScoringUnit, Score, State

@dataclass
class Message:
    '''
    The dataclass is an auxiliary generic component that
    is shared among the entire data-flow.
    The aim of the class is to make different components communicate
    through the data-passing of common object they all can manipulate.
    '''
    
    start_time   : float = 0
    end_time     : float = 0
    
    @property
    def elapsed_time(self) -> float:
        
        if not self.start_time:
            raise ValueError('Cannot compute elapsed time: start time not set')
        
        if not self.end_time:
            raise ValueError('Cannot compute elapsed time: end time not set')
        
        return self.end_time - self.start_time


@dataclass
class ZdreamMessage(Message):
    '''
    Message with attributes specific for ZdreamExperiment
    '''
    
    mask    : Mask = field(default_factory=lambda: np.array([]))
    '''
    Boolean mask associated to a set of stimuli indicating if they are
    synthetic of natural images. Defaults to empty array indicating absence 
	of natural images.
    '''
    
    label   : List[int] = field(default_factory=list)
    '''
    List of labels associated to the set of stimuli. Defaults to empty list.
    
    NOTE: Labels are only associated to natural images so they are
          they only refers to 'False' entries in the mask.
    '''
    
    n_group : int = 1
	# TODO: Write documentation
    '''
    
    '''
    
    masks_history      : List[Mask]                 = field(default_factory=list)
    codes_history      : List[Codes]                = field(default_factory=list)
    labels_history     : List[List[int]]            = field(default_factory=list)
    states_history     : List[State]         = field(default_factory=list)
    scores_gen_history : List[Score]         = field(default_factory=list)
    scores_nat_history : List[Score]         = field(default_factory=list)
    rec_units : Dict[str, RecordingUnit]            = field(default_factory=dict)
    scr_units : Dict[str, ScoringUnit]              = field(default_factory=dict)
    rf_maps   : Dict[Tuple[str, str], List[RFBox]]  = field(default_factory=dict)
    # TODO: Write documentation
    '''
    
    '''
    
    
    @property
    def codes(self) -> Codes:
        return self.codes_history[-1]
    
    @property
    def rec_layers(self) -> List[str]:
        return list(self.rec_units.keys())
    
    @property
    def scr_layers(self) -> List[str]:
        return list(self.scr_units.keys())
    
    @property
    def solution(self) -> Codes:
        '''
        Retrieve the code that produced the highest score
        from code scores history.

        :return: Best code score.
        :rtype: NDArray
        '''
        
        # Extract indexes for best scores
        flat_idx : np.intp  = np.argmax(self.scores_gen_history)
        best_gen, *best_idx = np.unravel_index(flat_idx, np.shape(self.scores_gen_history))
        
        # Extract the best code from generation and index
        best_code = self.codes_history[best_gen][best_idx]
        
        return best_code
    
    # --- STATISTICS ---    
    
    def _get_stats(self, scores: List[NDArray], synthetic: bool) -> Dict[str, Any]:
        '''
        Perform basic statics on scores, either associated to natural or synthetic images.

        :param scores: List of scores over generations.
        :type scores: List[NDArray]
        :param param: If statistics refer to synthetic images.
        :type param: bool
        :return: A dictionary containing different statics per score at different generations.
        :rtype: Dict[str, Any]
        '''
        
        # Extract indexes for best scores
        flat_idx : np.intp  = np.argmax(scores)
        best_gen, *best_idx = np.unravel_index(flat_idx, np.shape(scores))
        
        # Histogram indexes
        hist_idx : List[int]  = np.argmax(scores, axis=1)
        
        # We compute statics relative to scores
        stats = {
            'best_score' : scores[best_gen][best_idx],
            'curr_score' : scores[-1],
            'mean_shist' : np.array([np.mean(s) for s in scores]),
            'sem_shist'  : np.array([SEM(s)    for s in scores]),
            'best_shist' : [score[idx] for score, idx in zip(scores, hist_idx)],    
        }
        
        if synthetic:
            
            # Add information relative to codes
            stats_codes = {
                'best_code'  : self.solution,
                'curr_codes' : self.codes,
                'best_phist' : [codes[idx] for codes, idx in zip(self.codes_history, hist_idx)]
            }
            
            # Update stats dictionary
            stats.update(stats_codes)
            
        return stats
        
        
    @property
    def stats_gen(self) -> Dict[str, Any]:
        ''' Return statistics for synthetic codes '''
        return self._get_stats(scores=self.scores_gen_history, synthetic=True)
    
    @property
    def stats_nat(self) -> Dict[str, Any]: 
        ''' Return statistics for natural images '''
        return self._get_stats(scores=self.scores_nat_history, synthetic=False)

    