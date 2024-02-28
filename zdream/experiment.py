from os import path
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from .logger import Logger

from .generator import Generator
from .utils.model import Codes, Mask, Message, Stimuli, StimuliScore, SubjectState
from .optimizer import Optimizer
from .scores import Scorer
from .subject import InSilicoSubject
from .utils.misc import default, save_json, stringfy_time


@dataclass
class ExperimentConfig:
    '''
    The dataclass serves as a configuration for the Xdream Experiment, 
    it specifies an instance for the four main entities involved in the experiment.
    '''
    
    generator: Generator
    ''' Generator instance. '''
    
    subject: InSilicoSubject
    ''' Subject instance. '''
    
    scorer: Scorer
    ''' Scorer instance. '''
    
    optimizer: Optimizer
    ''' Optimize instance. '''
    
    iteration: int
    ''' Number of iteration to perform in the experiment. '''
    
    logger: Logger
    ''' Logger instance to log information, warnings and errors.'''
    
    mask_generator: Callable[[int], Mask | None] | None = None
    ''' 
    Function for mask generation. The boolean mask discriminates between synthetic
    and natural images in the stimuli. A new mask is potentially generated at 
    each iteration for a varying number of synthetic stimuli: the function
    maps their number to a valid mask (i.e. as many True values as synthetic stimuli).
    
    NOTE Mask generator defaults to None, that must be handled by Experiment class.
    '''
    
    data: Dict[str, Any] | None = None  
    '''
    General purpose attribute to allow the inclusion of any additional type of information.
    '''
    
    
class Experiment(ABC):
    '''
    This class implements the main pipeline of the Xdream experiment providing
    a granular implementation of the data flow.
    '''

    _EXPERIMENT_NAME = "Experiment"
    '''
    Experiment title. All experiment outputs will be
    saved in a sub-directory with its name.
    '''

    @classmethod
    @abstractmethod
    def from_config(cls, conf : str | dict[str, Any]) -> 'Experiment':
        pass
    
    def __init__(
            self, 
            config: ExperimentConfig,
            version: str = 'experiment'
        ) -> None:
        '''
        The constructor extract the terms from the configuration object.

        :param config: Experiment configuration.
        :type config: ExperimentConfig
        :param version: Name identifier for the experiment version, defaults to 'experiment'.
        :type version: str, optional
        :param param_config: Dictionary 
        :type version: Dict[str, Any]
        '''
        
        # Experiment version
        self._version = version
        
        # Configuration attributes
        self._generator      = config.generator
        self._subject        = config.subject
        self._scorer         = config.scorer
        self._optimizer      = config.optimizer
        self._logger         = config.logger
        self._iteration      = config.iteration
        
        # Defaults mask generator to a None mask, which is handled 
        # by the generator as a synthetic-only stimuli.
        self._mask_generator = default(config.mask_generator, lambda x: None)

        # Param config
        self._param_config: Dict[str, Any] = dict()

    def _set_param_configuration(self, param_config: Dict[str, Any]):
        '''
        Set the parameter configuration file for the experiment

        :param Dict: Parameter configuration dictionary
        :type Dict: Dict[str, Any])
        '''

        # NOTE: The method is private and separated from the `__init__()` method
        #       to make the classmethod `from_config` the only one allowed to
        #       set the parameter.

        self._param_config = param_config

        
    # --- PROPERTIES --- 
        
    # NOTE: Property methods override in experiment subclasses is crucial for
    #       activating specific behaviors of Generator, InSilicoSubject, Scorer 
    #       and Optimizer subclasses; the generic Experiment class only uses the 
    #       abstract classes along with their predefined methods and attributes. 
    #
    #       To activate subclass-specific attributes or methods, the corresponding
    #       property must be overridden casting to the specific subclass type.
    #       
    #       Example:
    #
    #       @property
    #       def generator(self) -> GeneratorSubclass: return cast(GeneratorSubclass, self._generator)
    #
    #       def foo(self): self.generator.method_subclass_specific()
        
    @property
    def generator(self) -> Generator:       return self._generator
    
    @property
    def subject(self)   -> InSilicoSubject: return self._subject
    
    @property
    def scorer(self)    -> Scorer:          return self._scorer
    
    @property
    def optimizer(self) -> Optimizer:       return self._optimizer

    @property
    def target_dir(self) -> str:
        return self._logger.target_dir
    
    # --- DATAFLOW METHODS ---
    
    # The following methods implements the main experiment data pipeline:
    #
    # Codes --Generator--> Stimuli --Subject--> SubjectState --Scorer--> StimuliScore --Optimizer--> Codes
    #
    # Along with component-specific data, a Message travels all along the pipeline. 
    # It allows for a generic data passing between components to communicate an information 
    # shared to entire system (e.g. the mask indicating synthetic and natural images).
    #
    # NOTE Experiment subclasses that intend to manipulate the data in the pipeline must
    #      override the specific function. The overridden method is in general advised not to
    #      redefine the default behavior but to redirect to it using the `super` keyboard. 

    def _codes_to_stimuli(self, codes: Codes) -> Tuple[Stimuli, Message]:
        '''
        The method uses the generator that maps input codes to visual stimuli.
        It uses the mask generator to generate the mask that interleaves
        synthetic and natural stimuli, and includes it in the Message.

        :param codes: Codes representing visual stimuli in the latent optimization space.
        :type codes: Codes
        :return: Generated visual stimuli and a Message
        :rtype: Tuple[Stimuli, Message]
        '''
        
        # Generate the mask based on the number of codes 
        # produced by the optimizer.
        mask = self._mask_generator(self.optimizer.n_states)
        
        return self.generator(codes=codes, mask=mask)
    
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        '''
        The method collects the Subject responses to a presented set of stimuli.

        :param data: Set of visual stimuli and a Message
        :type data: Tuple[Stimuli, Message]
        :return: Subject responses to the presented Stimuli and a Message.
        :rtype: Tuple[SubjectState, Message]
        '''
        
        return self.subject(data=data)
    
    def _sbj_state_to_stm_score(self, data: Tuple[SubjectState, Message]) -> Tuple[StimuliScore, Message]:
        '''
        The method evaluate the SubjectResponse in light of a Scorer logic.

        :param data: Subject responses to visual stimuli and a Message
        :type data: Tuple[SubjectState, Message]
        :return: A score for each presented stimulus.
        :rtype: Tuple[StimuliScore, Message]
        '''
        
        return self.scorer(data=data)
    
    def _stm_score_to_codes(self, data: Tuple[StimuliScore, Message]) -> Codes:
        '''
        The method uses the scores for each stimulus to optimize the images
        in the latent coded space, resulting in a new set of codes.
        
        NOTE: The optimization only works with synthetic images so the optimizer
              is asked to filter out natural ones using mask information in the message. 
        
        NOTE: Depending on the specific Optimizer implementation the number of 
              codes at each iteration is possibly varying.

        :param data: Score associated to the score and a Message 
        :type data: Tuple[StimuliScore, Message]
        :return: New optimized set of codes.
        :rtype: Codes
        '''
        
        return self.optimizer.step(data=data)
    
    # --- RUN ---
    
    # The general experiment pipeline is broken in different subfunctions
    # The granularity helps in the definition of a new experiment by
    # the modification of a specific component in the overall process.
    
    # NOTE: The overriding of the function is in general suggested not to redefine
    #       the default behavior but to redirect to it using `super` keyboard.
    
    def _init(self):
        '''
        The method is called before running the actual experiment.
        The default version logs the attributes of the components.
        It is supposed to contain all preliminary operations such as initialization.
        '''

        # Create experiment directory
        self._logger.create_target_dir()

        # Save and log parameters
        if self._param_config:
            
            # Log
            self._logger.info(f"")
            self._logger.info(f"Parameters:")
            max_key_len = max(len(key) for key in self._param_config.keys()) # for padding
            for k, v in self._param_config.items():
                self._logger.info(f'{k:<{max_key_len}}:   {v}')

            # Save
            config_param_fp = path.join(self.target_dir, 'params.json')
            self._logger.info(f"Saving param configuration to: {config_param_fp}")
            save_json(data=self._param_config, path=config_param_fp)

        # Components
        self._logger.info(f"")
        self._logger.info(f"Components:")
        self._logger.info(mess=f'Generator: {self.generator}')
        self._logger.info(mess=f'Subject:   {self.subject}')
        self._logger.info(mess=f'Scorer:    {self.scorer}')
        self._logger.info(mess=f'Optimizer: {self.optimizer}')
        self._logger.info(f"")

        
    def _finish(self):
        '''
        The method is called at the end of the actual experiment.
        The default version logs the experiment elapsed time.
        It is supposed to perform operation of experiment results.
        '''

        str_time = stringfy_time(sec=self._elapsed_time)
        self._logger.info(mess=f"Experiment finished successfully. Elapsed time: {str_time} s.")
    
    def _progress_info(self, i: int) -> str:
        '''
        The method returns the information to log in the progress at each iteration.
        Default version returns the progress percentage.
        
        NOTE: The function is separated from the `_progress` function to allow to concatenate
              default progress information with potential new one in a single line.

        :param i: Current iteration number.
        :type i: int
        :return: Progress information as a string.
        :rtype: str
        '''
        
        progress = f'{i:>{len(str(self._iteration))}}/{self._iteration}'
        perc     = f'{i * 100 / self._iteration:>5.2f}%'
        
        return f'{self._EXPERIMENT_NAME}[{self._version}]: [{progress}] ({perc})'
    
    def _progress(self, i: int):
        '''
        The method is called at the end of any experiment iteration.
        The default version logs the default progress information.

        :param i: Current iteration number.
        :type i: int
        '''
        self._logger.info(self._progress_info(i=i))
        
    def _run(self):
        '''
        The method implements the core of the experiment.
        It initializes random codes and iterates combining 
        the dataflow in the pipeline for the configured number 
        of iterations.
        '''

        self._logger.info("Running...")
        
        # Codes initialization
        codes = self.optimizer.init()
        
        for i in range(self._iteration):
            
            stimuli   = self._codes_to_stimuli(codes)
            sbj_state = self._stimuli_to_sbj_state(stimuli)
            stm_score = self._sbj_state_to_stm_score(sbj_state)            
            codes     = self._stm_score_to_codes(stm_score)

            self._progress(i=i)
    
    def run(self):
        '''
        The method implements the experiment logic by combining
        `init()`, `run()` and `finish()` methods. 
        It computes the total elapsed time of the experiment that can
        be accessed in the `finish()` method.
        
        NOTE: It is the only public one and so the only one supposed
              to be used from the caller.
        '''
        
        self._init()
        
        start_time = time.time()
        
        self._run()
        
        end_time = time.time()
        
        self._elapsed_time = end_time - start_time
        
        self._finish()

class MultiExperiment:
    '''
    
    '''
    
    def __init__(
        self,
        experiment : Experiment,
        base_config   : Dict[str, Any],
        search_config : Dict[str, Tuple[Any]], 
    ) -> None:
        
        # Check that provided search configuration argument
        # share the same length as length defines the number
        # of experiments to run
        err_msg = \
        '''
        Provided search configuration files have conflicting number
        of experiments runs (length of dictionary entries). Please
        provide coherent number of experiment runs.
        '''
        values = list(search_config.values())
        assert all([len(v) == values[0] for v in values]), err_msg
        
        self.Exp = experiment
        self.base_config = base_config
        
        # Convert the search configuration from 
        keys, vals = search_config.keys(), search_config.values()
        self.search_config : list[dict[str, Any]] = [
            {k : v for k, v in zip(keys, V)}
            for V in zip(*vals)
        ]

    def run(self):
        for conf in self.search_config:
            exp_config = {**self.base_config, **conf}
            exp_config = Namespace(**exp_config)
            
            exp = self.Exp.from_config(exp_config)
            
            exp.run()

