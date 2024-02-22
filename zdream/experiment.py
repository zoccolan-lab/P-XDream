from abc import ABC
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Tuple
from tqdm import tqdm


from .generator import Generator
from .optimizer import Optimizer
from .scores import Scorer
from .subject import InSilicoSubject
from .utils import Codes, Logger, Message, Stimuli, SubjectScore, SubjectState, default

@dataclass
class ExperimentConfig:
    
    generator: Generator
    subject: InSilicoSubject
    scorer: Scorer
    optimizer: Optimizer
    logger: Logger
    num_gen: int
    mask_generator: Callable[[int], List[bool] | None] | None = None
    data: Dict[str, Any] | None = None  # Any possible additional data
    
    
class Experiment:
    
    def __init__(self, config: ExperimentConfig,  name: str = 'Zdream') -> None:
        
        self._name = name
        
        self._generator       = config.generator
        self._subject         = config.subject
        self._scorer          = config.scorer
        self._optimizer       = config.optimizer
        self._logger          = config.logger
        self._num_gen         = config.num_gen
        
        self.mask_generator = default(config.mask_generator, lambda x: None)
        
    # --- PROPERTIES --- 
        
    # NOTE: The following property methods are fundamental to use specific behaviors of subclasses
    #       In the case specific methods of subclasses are required, the corresponding property must
    #        be overridden casting to the specific type 
        
    @property
    def generator(self) -> Generator: return self._generator
    
    @property
    def subject(self) -> InSilicoSubject: return self._subject
    
    @property
    def scorer(self) -> Scorer: return self._scorer
    
    @property
    def optimizer(self) -> Optimizer: return self._optimizer
    
    # --- PIPELINE METHODS ---

    def _codes_to_stimuli(self, codes: Codes) -> Tuple[Stimuli, Message]:
        
        mask = self.mask_generator(self.optimizer.n_states)
        
        return self.generator(codes=codes, mask=mask)
    
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        
        return self.subject(data=data)
    
    def _sbj_state_to_sbj_score(self, data: Tuple[SubjectState, Message]) -> Tuple[SubjectScore, Message]:
        
        return self.scorer(data=data)
    
    def _sbj_score_to_codes(self, data: Tuple[SubjectScore, Message]) -> Codes:
        
        return self.optimizer.step(data=data)
    
    # --- LOGGING UTILS ---
    
    def log_attributes(self):
        
        self._logger.info(mess=f'Generator: {self.generator}')
        self._logger.info(mess=f'Subject:   {self.subject}')
        self._logger.info(mess=f'Scorer:    {self.scorer}')
        self._logger.info(mess=f'Optimizer: {self.optimizer}')
            
    def progress_info(self, gen: int) -> str:
        return f'Generations: [{gen:>{len(str(self._num_gen))}}/{self._num_gen}] ({gen * 100 / self._num_gen:>5.2f}%)'
    
    
    # --- RUN ---
    
    def _init(self):
        self._logger.info(mess=f"Running experiment {self._name} with {self._num_gen} generations")
        self.log_attributes()
        
    def _finish(self):
        self._logger.info(mess=f"Experiment finished successfully. Elapsed time: {self._elapsed_time} s.")
    
    def _progress(self, gen: int):
        self._logger.info(self.progress_info(gen=gen))
        
    def _run(self):
        
        self._logger.info("Running...")
        
        codes = self.optimizer.init()
        
        for gen in range(self._num_gen):
            
            stimuli   = self._codes_to_stimuli(codes)
            sbj_state = self._stimuli_to_sbj_state(stimuli)
            sbj_score = self._sbj_state_to_sbj_score(sbj_state)            
            codes     = self._sbj_score_to_codes(sbj_score)

            self._progress(gen=gen)
    
    def run(self):
        
        self._init()
        
        start_time = time.time()
        
        self._run()
        
        end_time = time.time()
        
        self._elapsed_time = end_time - start_time
        
        self._finish()





