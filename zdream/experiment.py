'''
This file contains the main classes for running experiments in the Zdream framework.
It consists of two different kind of running experiments:
- Single experiment: a single instance of an experiment class.
- Multi experiment: multiple instances of the same experiment class with different hyperparameters.
'''


from __future__ import annotations

import os
from os import path
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress

from zdream.utils.parameters import ArgParam, ArgParams, ParamConfig, Parameter
from zdream.utils.dataset import NaturalStimuliLoader

from .utils.logger import DisplayScreen, Logger, SilentLogger
from .generator import Generator
from .utils.message import Message, ParetoMessage, ZdreamMessage
from .optimizer import Optimizer
from .scorer import Scorer
from .subject import InSilicoSubject
from .utils.io_ import read_json, save_json, store_pickle
from .utils.types import Codes, Stimuli, Scores, States
from .utils.misc import flatten_dict, load_npy_npz, overwrite_dict, stringfy_time


# --- EXPERIMENT ABSTRACT CLASS ---

class Experiment(ABC):
    '''
    Generic class implementing an Experiment to run.
    
    It implements a generic data flow for experiment execution based on
    three steps: `init`, `run` and `finish`.
    
    It also provides an automatic foldering organization for storing 
    experiment data building a three-level hierarchy:
    - Title: containing all experiment results of the data run with the same Experiment class
    - Name: name of the specific experiment for a specific investigation, that can also consists of multiple runs.
    - Version: version of the same experiment run with a different set of hyperparameters.
    
    The class can be instantiated in two modalities:
    - 1) By an explicit instantiations of a class.
    - 2) From a configuration.
    '''

    EXPERIMENT_TITLE = "Experiment"
    
    # --- INIT ---
    
    def __init__(
        self,    
        name   : str = 'experiment',
        logger : Logger = SilentLogger(),
        data   : Dict[str, Any] = dict(),
    ) -> None:
        
        '''
        Initialization of the experiment with a name and a logger.

        :param name: Name identifier for the experiment version, defaults to 'experiment'.
        :type name: str, optional
        :param logger: Logger instance to log information, warnings and errors;
            to handle directory paths and to display screens.
        :type logger: Logger
        :param data: General purpose attribute to allow the inclusion of any additional type of information.
        :type data: Dict[str, Any]
        '''
        
        # Experiment name
        self._name = name
        
        # Logger
        self._logger = logger

        # NOTE: `Data` input is not used in the default version.
        #       In general we don't want to save as an attribute the data dictionary,
        #       but to extract it values and to use them in potentially different ways.
        #       It's use in delegated to subclasses.
        

    def _set_param_configuration(self, param_config: ParamConfig):
        '''
        Set the parameter configuration as experiment attribute

        :param Dict: Parameter configuration dictionary
        :type Dict: Dict[str, Any])
        '''

        # NOTE: The method is private and separated from the `__init__()` method
        #       to make the classmethod `from_config` the only one allowed to
        #       set the parameter.

        self._param_config: ParamConfig = param_config
    
    
    # --- CONFIGURATION ---
    
    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> 'Experiment':
        '''
        Factory to instantiate an experiment from input parsing
        The configuration is supposed to contain `config` key which
        specify the default configuration location 

        :param args: Terminal parsed argument.
        :type args: Dict[str, Any]
        :return: Experiment instance.
        :rtype: Experiment
        '''
        
        # Read default configuration
        json_conf = read_json(args['config'])
        
        # Filter given arguments
        args_conf = {k : v for k, v in args.items() if v}

        # Combine set and default hyperparameters
        full_conf = overwrite_dict(json_conf, args_conf)
            
        return cls.from_config(conf=full_conf)

    @classmethod
    def from_config(cls, conf : ParamConfig) -> 'Experiment':
        '''
        Factory to instantiate an experiment from a hyperparameter configuration.

        :param conf: Experiment hyperparameter configuration
        :type conf: Dict[str, Any]
        :return: Experiment instantiated with proper configuration
        :rtype: Experiment
        '''
        
        # NOTE: The method only occupies to call the private `_from_config()` function
        #       and to set the configuration as experiment attribute.
        #       The method is not really supposed to be overridden, but just the private one.
        
        experiment = cls._from_config(conf=conf)
        
        experiment._set_param_configuration(param_config=conf)

        return experiment

    @classmethod
    @abstractmethod
    def _from_config(cls, conf : ParamConfig) -> 'Experiment':
        '''
        Method to instantiate an experiment from a hyperparameter configuration.

        :param conf: Experiment hyperparameter configuration
        :type conf: Dict[str, Any]
        :return: Experiment instantiated with proper configuration
        :rtype: Experiment
        '''
        pass
    
    # --- STRING REPRESENTATION ---
    
    def __str__ (self)  -> str: return f'{self.EXPERIMENT_TITLE}[{self._name}]'
    ''' Return a string representation of the experiment. '''
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the experiment.'''

    # --- PROPERTIES ---

    @property
    def dir(self) -> str:
        ''' 
        Experiment target directory is the Logger one.
        '''
        
        return self._logger.dir
    
    @property
    def _components(self) -> List[Tuple[str, Any]]:
        ''' 
        List of experiment components with their name.
        The method is using for logging purposes.
        '''
        
        return []
    
    # --- RUN ---
    
    # The general experiment pipeline is broken in different subfunctions
    # The granularity helps in the definition of a new experiment by
    # the modification of a specific component in the overall process.
    #
    # The full pipeline serves as a generic data structure called `Message`
    # which is passed ubiquitous in the data flows and serves as a generic
    # message passing information shared across the overall system.
    
    # NOTE: The overriding of the function is in general suggested not to redefine
    #       the default behavior but to redirect to it using `super` keyboard.
    
    def _init(self) -> Message:
        '''
        The method is called before running the actual experiment.
        The default version:
        - create experiments directory
        - logs the experiment parameters (if present) and the components.
        - generate the initial message
        
        It is supposed to contain all preliminary operations such as initialization.
        '''

        # Create experiment directory
        self._logger.create_dir()

        # Save and log parameters
        if hasattr(self, '_param_config'):

            flat_dict = ArgParam.argconf_to_json(self._param_config)
            
            # Log
            self._logger.info(f"")
            self._logger.info(mess=str(self))
            self._logger.info(f"Parameters:")

            max_key_len = max(len(key) for key in flat_dict.keys()) + 1 # for padding

            for k, v in flat_dict.items():
                k_ = f"{k}:"
                self._logger.info(mess=f'{k_:<{max_key_len}}   {v}')
            self._logger.info(mess=f'')

            # Save
            config_param_fp = path.join(self.dir, 'params.json')
            self._logger.info(f"Saving param configuration to: {config_param_fp}")
            save_json(data=ArgParam.argconf_to_json(self._param_config), path=config_param_fp)

        # Log components
        if self._components:
            
            self._logger.info(f'')
            self._logger.info(f'Components:')
            max_key_len = max(len(key) for key, _ in self._components) + 1 # for padding 
            for k, v in self._components:
                k_ = f"{k}:"
                self._logger.info(mess=f'{k_:<{max_key_len}}   {v}')
            self._logger.info(mess=f'')
        
                
        # Generate an initial message containing the start time
        msg = Message(
            start_time = time.time()
        )
        
        return msg

        
    def _finish(self, msg : Message) -> Message:
        '''
        The method is called at the end of the actual experiment.
        The default version logs the experiment elapsed time.
        
        It is supposed to perform operation of experiment results.
        '''
        
        # Set experiment end time
        msg.end_time = time.time()

        # Log total elapsed time
        str_time = stringfy_time(sec=msg.elapsed_time)
        self._logger.info(mess=f"Experiment finished successfully. Elapsed time: {str_time}.")
        self._logger.info(mess="")

        # NOTE: The method is also supposed to close logger screens
        #       However this is not implemented in the default version 
        #       because it may be possible to keep screen active for
        #       other purposes, so the logic is lead to subclasses.
        
        return msg
    
    @abstractmethod
    def _run(self, msg : Message) -> Message:
        '''
        The method implements the core of the experiment.
        Which is lead to subclasses
        '''
        pass
    
    def run(self) -> Message:
        '''
        The method implements the experiment logic by combining
        `_init()`, `_run()` and `_finish()` methods.
        
        :return: Message produced by the experiment
        :rtype: Message
        '''
        
        msg = self._init()
        msg = self._run(msg)
        msg = self._finish(msg)
        
        self._logger.close()
        
        return msg

# --- ZDREAM ---

@dataclass
class ZdreamExperimentState:
    '''
    Dataclass collecting the main data structure describing the 
    state of an Zdream experiment across generation and allows 
    to dump states to disk.

    The class can be instantiated either by providing an `ZdreamExperiment` object
    or by the folder path containing dumped states.
    '''
    
    FILES = [
        ('mask',       'npy'),
        ('labels',     'npy'),
        ('states',     'npz'),
        ('codes',      'npy'),
        ('scores_gen', 'npy'),
        ('scores_nat', 'npy'),
        ('rec_units',  'npz'),
        ('scr_units',  'npz'),
        ('rf_maps',    'npz'),]

    # NOTE: We don't use the defined type aliases `Codes`, `State`, ...
    #       as they refer to a single step in the optimization process, while
    #       the underlying datastructures consider an additional first dimension
    #       in the Arrays relative to the generations.
    
    # NOTE: All states can take None value, that refers to the absence of the state

    mask:       NDArray[np.bool_]                        | None  # [generations x (n_gen + n_nat)]
    labels:     NDArray[np.int32]                        | None  # [generations x n_nat]
    codes:      NDArray[np.float32]                      | None  # [generations x n_gen x code_len] 
    scores_gen: NDArray[np.float32]                      | None  # [generations x n_gen]
    scores_nat: NDArray[np.float32]                      | None  # [generations x n_nat]
    states:     Dict[str, NDArray[np.float32]]           | None  # layer_name: [generations x (n_gen + n_nat) x layer_dim]
    rec_units:  Dict[str, NDArray[np.int32]]             | None  # layer_name: [recorded units]
    scr_units:  Dict[str, NDArray[np.int32]]             | None  # layer_name: [scores activation indexes]
    rf_maps:    Dict[Tuple[str, str], NDArray[np.int32]] | None  # (layer mapped, layer of mapping units): receptive fields
    
    @classmethod
    def from_msg(cls, msg : ZdreamMessage) -> 'ZdreamExperimentState':
        
        # Collate histories into a single data bundle
        states = {
            key: np.stack([state[key] for state in msg.states_history]) if msg.states_history else np.array([])
            for key in msg.rec_layers
        }
        
        # Stack together dictionary components
        rec_units = {k: np.stack(v).T if v else np.array([]) for k, v in msg.rec_units.items()}
        scr_units = {k: np.array(v)                          for k, v in msg.scr_units.items()}
        rf_maps   = {k: np.array(v, dtype=np.int32)          for k, v in msg.rf_maps.  items()}
        
        # Create experiment instance by stacking histories in a generation-batched array
        return ZdreamExperimentState(
            mask       = np.stack(msg.masks_history),
            labels     = np.stack(msg.labels_history),
            states     = states,
            codes      = np.stack(msg.codes_history),
            scores_gen = np.stack(msg.scores_gen_history),
            scores_nat = np.stack(msg.scores_nat_history),
            rec_units  = rec_units,
            scr_units  = scr_units,
            rf_maps    = rf_maps 
        )

    @classmethod
    def from_path(cls, in_dir: str, logger: Logger = SilentLogger()):
        '''
        Load an experiment state from a folder where the experiment
        was dumped. If raises a warning for not present states.

        :param in_dir: Directory where states are dumped.
        :type in_dir: str
        :param logger: Logger to log i/o information. If not specified
            a `SilentLogger` is used. 
        :type logger: Logger | None, optional
        '''
        loaded = load_npy_npz(in_dir = in_dir, fnames = cls.FILES, logger = logger)
        cls_init_args = {name: loaded[name] for name, _ in cls.FILES}
        return cls(**cls_init_args)


    def dump(self, out_dir: str, logger: Logger = SilentLogger(), store_states: bool = False):
        '''
        Dump experiment state to an output directory.
        NOTE:   Dumping the subject state is optional as typically memory demanding;
                This is why is only for this variable is present an flag to indicate if to store them.

        :param out_dir: Directory where to dump states.
        :type out_dir: str
        :param logger: Logger to log i/o information. If not specified a `SilentLogger` is used. 
        :type logger: Logger | None, optional
        :param store_states: If to dump subject states.
        :type store_states: bool
        '''
        
        loads = self.FILES
        
        if store_states:
            loads.append(('states','npz'))

        # Output directory
        logger.info(f'Dumping experiment to {out_dir}')
        os.makedirs(out_dir, exist_ok=True)

        for name, ext in loads:

            # File path
            fp = path.join(out_dir, f'{name}.{ext}')

            # Saving function depending on file extension
            match ext:
                case 'npy': save_fun = np.save
                case 'npz': save_fun = lambda file, x: np.savez(file, **x)

            # The checking function checks if the data structure contains no information
            # and it's specific to each type of data. In case of no information it is not stored.
            match name:
                case 'labels' | 'scores_nat':  check_fn = lambda x : x.size
                case 'rf_maps':                check_fn = lambda x : len(x)
                case 'states':                 check_fn = lambda x : all([v.size for v in x.values()])
                case _:                        check_fn = lambda _ : True
                #TODO @Tau
                #add checks for new info from ParetoExperimentState
            # Get specific state 
            state = self.__getattribute__(name)
            
            # Save if actual information
            if check_fn(state):
                logger.info(f"> Saving {name} history from {fp}")
                save_fun(fp, state)
            
            # Warning if no information
            else:
                logger.warn(f'> Attempting to dump {name}, but empty')
        
        logger.info(f'')

@dataclass
class ParetoExperimentState(ZdreamExperimentState):
    FILES = ZdreamExperimentState.FILES + [
    ('pf1_coords', 'npy'),
    ('layer_scores_gen_history', 'npz'),]
    
    pf1_coords: NDArray[np.int32]                          | None  # [n_pareto1 x n_coords]
    layer_scores_gen_history: Dict[str, NDArray[np.float32]] | None  # layer_name: [n_gen x pop_sz]
    
    @classmethod
    def from_msg(cls, msg : ParetoMessage) -> 'ParetoExperimentState':
        exp_state = super().from_msg(msg)
        
        layer_scores = {k:np.vstack(v) for k,v in msg.layer_scores_gen_history.items()}
        
        return ParetoExperimentState(
            mask       = exp_state.mask,
            labels     = exp_state.labels,
            states     = exp_state.states,
            codes      = exp_state.codes,
            scores_gen = exp_state.scores_gen,
            scores_nat = exp_state.scores_nat,
            rec_units  = exp_state.rec_units,
            scr_units  = exp_state.scr_units,
            rf_maps    = exp_state.rf_maps,
            pf1_coords = msg.Pfront_1,
            layer_scores_gen_history = layer_scores 
        )
        


class ZdreamExperiment(Experiment):
    '''
    This class implements the main pipeline of the Zdream experiment providing
    a granular implementation of the data flow.
    '''

    EXPERIMENT_TITLE = "ZdreamExperiment"
    
    # --- INIT ---

    def __init__(
        self,    
        generator:      Generator,
        subject:        InSilicoSubject,
        scorer:         Scorer,
        optimizer:      Optimizer,
        iteration:      int,
        logger:         Logger = SilentLogger(),
        nat_img_loader: NaturalStimuliLoader = NaturalStimuliLoader(),
        data:           Dict[str, Any] = dict(),
        name:           str = 'zdream-experiment'
    ) -> None:
        '''

        :param generator: Generator instance.
        :type generator: Generator.
        :param subject: InSilicoSubject instance.
        :type subject: InSilicoSubject.
        :param scorer: Scorer instance.
        :type optimizer: Optimizer.
        :param optimizer: Optimizer instance.
        :type generator: Generator.
        :param generator: Generator instance.
        :type generator: Generator.
        :param iteration: Number of iteration (generation steps) to perform in the experiment.
        :type iteration: int
        :param logger: Logger instance to log information, warnings and errors,
            to handle directory paths and to display screens.
        :type logger: Logger
        :param nat_img_loader: Loader for natural images to be used in the experiment.
            If not given it defaults to a trivial `NaturalStimuliLoader` that loads no natural image.
        :type nat_img_loader: NaturalStimuliLoader, optional
        :param data: General purpose attribute to allow the inclusion of any additional type of information.
        :type data: Dict[str, Any]
        :type config: ExperimentConfig
        :param name: Name identifier for the experiment version, defaults to 'experiment'.
        :type name: str, optional
        '''
        
        super().__init__(
            name=name,
            logger=logger,
            data=data
        )
        
        # Experiment name
        self._name = name
        
        # Configuration attributes
        self._generator      = generator
        self._subject        = subject
        self._scorer         = scorer
        self._optimizer      = optimizer
        self._logger         = logger
        self._iteration      = iteration
        self._nat_img_loader = nat_img_loader


    def _set_param_configuration(self, param_config: ParamConfig):
        '''
        Set the parameter configuration file for the experiment

        :param Dict: Parameter configuration dictionary
        :type Dict: Dict[str, Any])
        '''
        
        # NOTE: We remove from the configuration file 
        #       the display screens potentially present from
        #       multi-experiment configuration
        param_config.pop(ArgParams.DisplayScreens.value, None)
        
        super()._set_param_configuration(param_config=param_config)

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
    def iteration(self) -> int:             return self._iteration
    
    @property
    def components(self) -> List[Tuple[str, Any]]:
        '''
        List of experiment components with their name.
        '''
        
        return [
            ('Generator', self.generator),
            ('Subject',   self.subject),
            ('Scorer',    self.scorer),
            ('Optimizer', self.optimizer)
        ]
    
    @property
    def num_imgs(self) -> int: return self._optimizer.pop_size
    ''' 
    TODO @Paolo    
    '''
    
    # --- DATAFLOW METHODS ---
    
    # The following methods implements the main experiment data pipeline:
    #
    # Codes --[Generator]-> Stimuli --[Subject]-> State --[Scorer]-> Score --[Optimizer]-> Codes
    #
    # Along with component-specific data, a Message travels all along the pipeline. 
    # It allows for a generic data passing between components to communicate an information 
    # shared to entire system (e.g. the mask indicating synthetic and natural images).
    #
    # NOTE Experiment subclasses that intend to manipulate the data in the pipeline must
    #      override the specific function. The overridden method is in general advised not to
    #      redefine the default behavior but to redirect to it using the `super` keyboard.
    

    def _codes_to_stimuli(self, data: Tuple[Codes, ZdreamMessage]) -> Tuple[Stimuli, ZdreamMessage]:
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
        codes, msg = data
        
        # Produce synthetic images
        gen_img = self.generator(codes=codes)
        
        # Create synthetic stimuli
        num_gen_img, *_ = gen_img.shape
        nat_img, labels, mask = self._nat_img_loader.load_natural_images(num_gen_img=num_gen_img)
        
        # Interleave the synthetic and natural images
        stimuli = NaturalStimuliLoader.interleave_gen_nat_stimuli(
            gen_img=gen_img,
            nat_img=nat_img,
            mask=mask
        )
        
        # Update message info
        msg.masks_history .append(mask)
        msg.labels_history.append(labels)
        
        return stimuli, msg
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        The method collects the Subject responses to a presented set of stimuli.

        :param data: Set of visual stimuli and a Message
        :type data: Tuple[Stimuli, Message]
        :return: Subject responses to the presented Stimuli and a Message.
        :rtype: Tuple[State, Message]
        '''
        
        stimuli, msg = data
        
        # Subject step
        states = self.subject(stimuli=stimuli)
        
        # NOTE: By default we decide not not save subject states
        #       as they can be memory demanding if in terms of RAM memory
        #       Experiments who needs to save subject states needs to 
        #       override the method
        
        # msg.states_history.append(state)
    
        return states, msg
    
    def _states_to_scores(self, data: Tuple[States, ZdreamMessage]) -> Tuple[Scores, ZdreamMessage]:
        '''
        The method evaluate the SubjectResponse in light of a Scorer logic.

        :param data: Subject responses to visual stimuli and a Message
        :type data: Tuple[State, Message]
        :return: A score for each presented stimulus.
        :rtype: Tuple[Score, Message]
        '''
        
        states, msg = data
        
        # Scorer step
        scores = self.scorer.__call__(states=states)
        if scores is not None:
            # Update message scores history (synthetic and natural)
            msg.scores_gen_history.append(scores[ msg.mask])
            msg.scores_nat_history.append(scores[~msg.mask])
        
        return scores, msg
    
    def _scores_to_codes(self, data: Tuple[Scores, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        '''
        The method uses the scores for each stimulus to optimize the images
        in the latent coded space, resulting in a new set of codes.
        
        NOTE:   Depending on the specific Optimizer implementation the number of 
                codes at each iteration is possibly varying.

        :param data: Score associated to the score and a Message 
        :type data: Tuple[Score, Message]
        :return: New optimized set of codes.
        :rtype: Codes
        '''
        
        scores, msg = data
        
        # Filter scores only for synthetic images
        scores_ = scores[msg.mask]
        
        # Optimizer step
        codes = self.optimizer.step(scores=scores_)
    
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg
    
    # --- RUN ---
    
    # The general experiment pipeline is broken in different subfunctions
    # The granularity helps in the definition of a new experiment by
    # the modification of a specific component in the overall process.
    
    # NOTE: The overriding of the function is in general suggested not to redefine
    #       the default behavior but to redirect to it using `super` keyboard.
    
    def _init(self) -> ZdreamMessage:
        '''
        The method is called before running the actual experiment.
        It adds to the message the number of recorded units 
        '''

        msg = super()._init()
        
        # NOTE: We need to create a new specif message for Zdream
        msg_ = ZdreamMessage(
            start_time = msg.start_time,
            rec_units  = self.subject.target,
            scr_units  = self.scorer.scoring_units,
        )
        
        return msg_

        
    def _finish(self, msg : ZdreamMessage) -> ZdreamMessage:
        '''
        The method is called at the end of the actual experiment.
        It performs the dump of states history contained in the message.
        '''
        
        msg = cast(ZdreamMessage, super()._finish(msg=msg))

        # Dump
        state = ZdreamExperimentState.from_msg(msg=msg)
        
        state.dump(
            out_dir=path.join(self.dir, 'state'),
            logger=self._logger
        )

        # NOTE: The method is also supposed to close logger screens
        #       However this is not implemented in the default version 
        #       because it may be possible to keep screen active for
        #       other purposes, so the logic is lead to subclasses.
        
        return msg
    
    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:
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

        j = i + 1 # index is off-by-one
        
        progress = f'{j:>{len(str(self._iteration))}}/{self._iteration}'
        perc     = f'{j * 100 / self._iteration:>5.2f}%'
        
        return f'{self}: [{progress}] ({perc})'
    
    def _progress(self, i: int, msg : ZdreamMessage):
        '''
        The method is called at the end of any experiment iteration.
        The default version logs the default progress information.

        :param i: Current iteration number.
        :type i: int
        '''
        self._logger.info(self._progress_info(i=i, msg=msg))

        
    def _run_init(self, msg : ZdreamMessage) -> Tuple[Codes, ZdreamMessage]:
        '''
        Method called before entering the main for-loop across generations.
        It is responsible for generating the initial codes
        '''
        
        # Codes initialization
        codes = self.optimizer.init()
        
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg
        
    def _run(self, msg : ZdreamMessage) -> ZdreamMessage:
        '''
        The method implements the core of the experiment.
        It initializes random codes and iterates combining 
        the dataflow in the pipeline for the configured number 
        of iterations.
        '''

        self._logger.info("Running...")
        
        codes, msg = self._run_init(msg)
        
        for i in range(self._iteration):
            self._curr_iter = i
            stimuli, msg = self._codes_to_stimuli (data=(codes,   msg))
            states,  msg = self._stimuli_to_states(data=(stimuli, msg))
            scores,  msg = self._states_to_scores (data=(states,  msg))
            codes,   msg = self._scores_to_codes  (data=(scores,  msg))

            self._progress(i=i, msg=msg)
            
            if msg.early_stopping:
                self._logger.warn(f'Early stopping at iteration {i}')
                break
        
        return msg

# --- MULTI EXPERIMENT ---

class MultiExperiment:
    '''
    Class for running multiple versions of the same experiment
    varying the set of hyperparameters using a configuration file
    and a default hyperparameters settings.
    '''
    
    ''' Parameter flag to render screens'''
    
    # --- INIT ---
    
    def __init__(
        self,
        experiment      : Type['Experiment'],
        experiment_conf : Dict[ArgParam, List[Parameter]], 
        default_conf    : ParamConfig
    ) -> None:
        '''
        Create a list of experiments configurations by combining
        the provided experiment configuration with default ones.

        :param experiment: Experiment class for conducing multi-experiment run.
        :type experiment: Type['Experiment']
        :param experiment_conf: Configuration for the multi-experiment specifying a list
                                of values for a set of hyperparameters. 
                                All lists are expected to have the same length.        
        :param default_conf: Default configuration file for all hyperparameters not 
            explicitly provided in the experiment configuration.
        :type default_conf: Dict[str, Any]
        '''
        
        # Check that provided search configuration argument share the same 
        # length as length defines the number of experiments to run.
        lens = [len(v) for v in experiment_conf.values()]
        
        if len(set(lens)) > 1:
            err_msg =   f'Provided search configuration files have conflicting number of experiments runs: {lens}.'\
                        f'Please provide coherent number of experiment runs.'
            raise ValueError(err_msg)
        
        # Save experiment class
        self._Exp = experiment
        
        # Convert the search configuration and combine with default values 
        keys, vals = experiment_conf.keys(), experiment_conf.values()

        self._search_config : List[ParamConfig] = [
            overwrite_dict(
                default_conf, 
                {k : v for k, v in zip(keys, V)}
            )
            for V in zip(*vals)
        ]

        # Set the logger with proper checks
        self._logger = self._set_logger()
        

    def _set_logger(self) -> Logger:
        '''
        Utility for the `_init_()` function to set the logger.
        Set a multi-experiment level logger and checks name consistency between experiments
        '''

        # Check single experiment name
        exp_names = [conf[ArgParams.ExperimentName.value] for conf in self._search_config]
        
        if len(set(exp_names)) > 1:
            err_msg = f'Multirun expects a unique experiment name, but multiple found {set(exp_names)}'
            raise ValueError(err_msg)

        # Check multiple output directories
        out_dirs = [conf[ArgParams.OutputDirectory.value] for conf in self._search_config]
        
        if len(set(out_dirs)) > 1:
            err_msg = f'Multirun expects a unique output directory, but multiple found {set(out_dirs)}'
            raise ValueError(err_msg)

        # Set multi-experiment name
        self._name    = exp_names[0]

        return self._logger_type(
            path = {
                'out_dir': str(out_dirs[0]),
                'title'  : str(self._Exp.EXPERIMENT_TITLE),
                'name'   : str(self._name)
            }
        )
        
        
    @property
    def _logger_type(self) -> Type[Logger]:
        '''
        Returns multi-experiment level logger type. 
        NOTE:   Subclasses that want to change the logger type only need
                to override this property.
        '''
        return Logger
    
    @classmethod
    def from_args(
        cls,
        arg_conf: ParamConfig,
        default_conf: ParamConfig,
        exp_type: Type[Experiment]
    ):
        '''
        Initialize a multi-experiment from hyperparameter configuration.
        The configuration is supposed to contain `config` key which
        specify the default configuration location 
        '''

        # Get type dictionary for casting
        # dict_type = {k: type(v) for k, v in flatten_dict(json_conf).items()}

        # Config from command line
        args_conf = {}

        # Keep track of argument lengths
        observed_lens = set()

        # Loop on input arguments
        for arg, value in arg_conf.items():

            # Get typing for cast
            type_cast = arg.type

            # Split input line with separator # and cast
            args_conf[arg] = [
                type_cast(a.strip()) for a in str(value).split('#')
            ]

            # Add observed length if different from one
            n_arg = len(args_conf[arg])
            
            if n_arg != 1:
                observed_lens.add(n_arg)

        # Check if multiple lengths
        if len(observed_lens) > 1:
            raise SyntaxError(f'Multiple argument with different lengths: {{observed_lens}}')

        # Check for no multiple args specified
        if len(observed_lens) == 0:
            raise SyntaxError(f'No multiple argument was specified.')

        # Adjust 1-length values
        n_args = list(observed_lens)[0]
        args_conf = {k : v * n_args if len(v) == 1 else v for k, v in args_conf.items()}
        
        return cls(
            experiment=exp_type,
            experiment_conf=args_conf,
            default_conf=default_conf
        )
        
    # --- MAGIC METHODS ---

    def __len__ (self) -> int: return len(self._search_config)
    def __str__ (self) -> str: return f'MultiExperiment[{self._name}]({len(self)} versions)'
    def __repr__(self) -> str: return str(self)

    # --- PROPERTIES ---

    @property
    def target_dir(self) -> str: return self._logger.dir
    
    # --- UTILITIES ---

    def _get_display_screens(self) -> List[DisplayScreen]:
        '''
        Returns the list of screens used in the experiment.
        
        NOTE: This is made to make all experiments share and reuse the same
        screen and prevent screen instantiation overhead and memory leaks.

        NOTE:   The method is not a property because it would be pre-executed to simulate
                attribute behavior even if not explicitly called and we want to avoid
                the creation of the `DisplayScreen` with no usage. 
        '''

        # In the default version we have no screen
        return []

    
    # --- RUN ---

    def _init(self):
        '''
        The method is called before running all the experiments.
        The default version logs the number of experiments to run, set the shared screens
        across involved experiments and initialize an empty dictionary where to store multi-run results.
        It is supposed to contain all preliminary operations such as initialization.
        '''

        # Initial logging
        self._logger.info(mess=f'RUNNING {self}')
        
        # NOTE: Handling screen turns fundamental in time
        #       overhead and memory during multi-experiment run.
        #       For this reason in the case at least one experiment 
        #       has the `render` option enabled we instantiate shared
        #       screens among all this experiment that are attached
        #       to the configuration dictionary.

        # Add screens if at least one experiment has `render` flag on
        if any(conf.get(ArgParams.Render.value, False) for conf in self._search_config):

            self._main_screen = DisplayScreen.set_main_screen() # keep reference
            self._screens     = self._get_display_screens()

            # Add screens to configuration to experiments with `render` flag on
            for conf in self._search_config:
                
                conf[ArgParams.CloseScreen.value] = False
                
                if conf.get(ArgParams.Render.value, False):
                    conf[ArgParams.DisplayScreens.value] = self._screens  # type: ignore
            
            self._search_config[-1][ArgParams.CloseScreen.value] = True
            
        # Set version name
        for i, conf in enumerate(self._search_config):
            conf[ArgParams.ExperimentVersion.value] = i
        

        # Data dictionary
        # NOTE: This is a general purpose data-structure where to save
        #       experiment results. It will be stored as a .pickle file.
        self._data: Dict[str, Any] = dict()

    def _progress(self, exp: Experiment, conf: ParamConfig, i: int, msg: Message):
        '''
        Method called after running a single experiment.
        In the default version it only logs the progress.

        :param exp: Instance of the run experiment.
        :type exp: Experiment
        :param i: Iteration of the multi-run.
        :type i: int
        '''

        j = i+1
        self._logger.info(mess=f'EXPERIMENT {j} OF {len(self)} RUN SUCCESSFULLY.')
        self._logger.info(mess=f'')

    def _finish(self):
        '''
        Method called after all experiments are run.
        It stores the results contained in the `data` dictionary if not empty.
        '''

        str_time = stringfy_time(sec=self._elapsed_time)
        self._logger.info(mess=f'ALL EXPERIMENT RUN SUCCESSFULLY. ELAPSED TIME: {str_time} s.')
        self._logger.info(mess=f'')

        # Save multi-run experiment as a .PICKLE file
        if self._data:
            out_fp = path.join(self.target_dir, 'data.pkl')
            self._logger.info(mess=f'Saving multi-experiment data to {out_fp}')
            store_pickle(data=self._data, path=out_fp)

    def _run(self):
        '''
        Run the actual multi-run by executing all experiments in 
        the provided configurations.
        '''
        
        # We need to specify end=''" as log message already ends with \n (thus the lambda function)
		# Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
        self._logger.set_progress_bar()
        
        with Progress(console=Logger.CONSOLE) as progress:

            for i, conf in progress.track(
                enumerate(self._search_config),
                total=len(self._search_config)
            ):  
                
                self._logger.info(mess=f'RUNNING EXPERIMENT {i+1} OF {len(self)}.')
                
                exp = self._Exp.from_config(conf=conf)
                
                msg = exp.run()
                
                self._progress(exp=exp, conf=conf, i=i, msg=msg)

    def run(self):
        '''
        The method implements the multi-experiment logic by combining
        `init()`, `run()` and `finish()` methods. 
        It computes the total elapsed time of the experiment that can
        be accessed in the `finish()` method.
        '''
        
        self._init()
        
        start_time = time.time()
        
        self._run()
        
        end_time = time.time()
        
        self._elapsed_time = end_time - start_time
        
        self._finish()
        
        self._logger.close()
