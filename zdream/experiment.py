from dataclasses import dataclass
from typing import Callable, List, Tuple
from tqdm import tqdm


from .generator import Generator
from .optimizer import Optimizer
from .scores import Scorer
from .subject import NetworkSubjectAbstract
from .utils import Codes, Logger, Message, Stimuli, SubjectScore, SubjectState, default

@dataclass
class ExperimentConfig:
    
    generator: Generator
    subject: NetworkSubjectAbstract
    scorer: Scorer
    optimizer: Optimizer
    logger: Logger
    num_gen: int
    mask_generator: Callable[[int], List[bool] | None] | None = None
    
    
class Experiment:
    
    def __init__(self, config: ExperimentConfig) -> None:
        
        self._generator       = config.generator
        self._subject         = config.subject
        self._scorer          = config.scorer
        self._optimizer       = config.optimizer
        self._logger          = config.logger
        self._num_gen         = config.num_gen
        
        self.mask_generator = default(config.mask_generator, lambda x: None)
        
    @property
    def generator(self) -> Generator: return self._generator
    
    @property
    def subject(self) -> NetworkSubjectAbstract: return self._subject
    
    @property
    def scorer(self) -> Scorer: return self._scorer
    
    @property
    def optimizer(self) -> Optimizer: return self._optimizer
        
    
    """
    What does a generator
    Stimuli, Msg <- Generator(Tensor, Msg)
    
    What does a subject
    SubjectState, Mgs <- Subject(Stimuli, Msg)
    
    What does a scorer
    SubjectScore, Msg <- Scorer(SubjectState, Msg)
    
    What does a Optimizer 
    Tensor <- Optimizer.step(SubjectScore, Msg)
    
    """
    
    def codes_to_stimuli(self, codes: Codes) -> Tuple[Stimuli, Message]:
        
        mask = self.mask_generator(self.optimizer.n_states)
        
        return self.generator(codes=codes, mask=mask)
    
    def stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        
        return self.subject(data=data)
    
    def sbj_state_to_sbj_score(self, data: Tuple[SubjectState, Message]) -> Tuple[SubjectScore, Message]:
        
        return self.scorer(data=data)
    
    def sbj_score_to_codes(self, data: Tuple[SubjectScore, Message]) -> Codes:
        
        return self.optimizer.step(data=data)
    
    def run(
        self, 
    ):
        
        codes = self.optimizer.init()
        
        for gen in tqdm(range(self._num_gen)):
            
            stimuli   = self.codes_to_stimuli(codes)
            sbj_state = self.stimuli_to_sbj_state(stimuli)
            sbj_score = self.sbj_state_to_sbj_score(sbj_state)            
            codes     = self.sbj_score_to_codes(sbj_score)




