from __future__ import annotations

from os import path
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import tkinter
from typing import Any, Dict, List, Tuple, Type, cast

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .utils.io_ import save_json, store_pickle

from .logger import Logger, MutedLogger

from .generator import Generator
from .utils.model import Codes, DisplayScreen, Mask, MaskGenerator, RFBox, Stimuli, Score, State
from .optimizer import Optimizer
from .scorer import Scorer
from .subject import InSilicoSubject
from .utils.misc import default, flatten_dict, overwrite_dict, stringfy_time
from .message import Message

@dataclass
class ExperimentState:
    '''
    Dataclass collecting the main data structure describing the 
    state of an experiment across generation and allows to dump
    states to disk.

    The class can be instantiated either by providing an `Experiment` object
    or by a folder path containing dumped states.
    '''

    # NOTE: We don't use the defined type aliases `Codes`, `State`, ...
    #       as they refer to a single step in the optimization process, while
    #       the underlying datastructures consider an additional first dimension
    #       in the Arrays relative to the generations.

    mask:       NDArray[np.bool_]                        | None  # [generations x (n_gen + n_nat)]
    labels:     NDArray[np.int32]                        | None  # [generations x n_nat]
    codes:      NDArray[np.float32]                      | None  # [generations x n_gen x code_len] 
    scores_gen: NDArray[np.float32]                      | None  # [generations x n_gen]
    scores_nat: NDArray[np.float32]                      | None  # [generations x n_nat]
    states:     Dict[str, NDArray[np.float32]]           | None  # layer_name: [generations x (n_gen + n_nat) x layer_dim]
    rec_units:  Dict[str, NDArray[np.int32]]             | None  # layer_name: recorded units
    scr_units:  Dict[str, NDArray[np.int32]]             | None  # layer_name: scores activation indexes
    rf_maps:  Dict[Tuple[str, str], NDArray[np.int32]]   | None  # (layer mapped, layer of mapping units): receptive fields
    
    @classmethod
    def from_msg(cls, msg : Message) -> 'ExperimentState':
        
        # Collate histories into a single data bundle
        states = {
            key: np.stack([state[key] for state in msg.states_history]) if msg.states_history else np.array([])
            for key in msg.rec_layers
        }
        
        rec_units = {k: np.stack(v).T if v else np.array([]) for k, v in msg.rec_units.items()}
        scr_units = {k: np.array(v)                          for k, v in msg.scr_units.items()}
        rf_maps   = {k: np.array(v, dtype=np.int32)          for k, v in msg.rf_maps.items()}
        
        return ExperimentState(
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
    def from_path(cls, in_dir: str, logger: Logger | None = None):
        '''
        Load an experiment state from a folder where the experiment
        was dumped. If raises a warning if not all states are present.

        :param in_dir: Directory where states are dumped.
        :type in_dir: str
        :param logger: Logger to log i/o information. If not specified
                       a `MutedLogger` is used. 
        :type logger: Logger | None, optional
        '''

        # Logger default
        logger = default(logger, MutedLogger())

        # File to load an their extension
        to_load = [
            ('mask',       'npy'),
            ('labels',     'npy'),
            ('states',     'npz'),
            ('codes',      'npy'),
            ('scores_gen', 'npy'),
            ('scores_nat', 'npy'),
            ('rec_units',  'npz'),
            ('scr_units',  'npz'),
            ('rf_maps',    'npz'),
        ]
        loaded = dict()

        logger.info(f'Loading experiment from {in_dir}')

        for name, ext in to_load:

            # File path
            fp = path.join(in_dir, f'{name}.{ext}')

            # Loading function depending on file extension
            match ext:
                case 'npy': load_fun = np.load
                case 'npz': load_fun = lambda x: dict(np.load(x))

            # Loading state
            if path.exists(fp):
                logger.info(f"> Loading {name} history from {fp}")
                loaded[name] = load_fun(fp)

            # Warning as the state is not present
            else:
                logger.warn(f"> Unable to fetch {fp}")
                loaded[name] = None

        return cls(
            mask       = loaded['mask'],
            labels     = loaded['labels'],
            states     = loaded['states'],
            codes      = loaded['codes'],
            scores_gen = loaded['scores_gen'],
            scores_nat = loaded['scores_nat'],
            rec_units  = loaded['rec_units'],
            scr_units  = loaded['scr_units'],
            rf_maps    = loaded['rf_maps']
        )


    def dump(self, out_dir: str, logger: Logger | None = None, store_states: bool = False):
        '''
        Dump experiment state to an output directory where it creates a proper `state` subfolder.
        Dumping the subject state is optional as typically memory demanding.

        :param out_dir: Directory where to dump states.
        :type out_dir: str
        :param logger: Logger to log i/o information. If not specified
                       a `MutedLogger` is used. 
        :type logger: Logger | None, optional
        '''
        
        # Logger default
        logger = default(logger, MutedLogger())

        # File to load an their extension
        to_load = [
            ('mask',       'npy'),
            ('labels',     'npy'),
            ('codes',      'npy'),
            ('scores_gen', 'npy'),
            ('scores_nat', 'npy'),
            ('rec_units',  'npz'),
            ('scr_units',  'npz'),
            ('rf_maps',    'npz'),
        ]
        
        if store_states:
            to_load.append(('states','npz'))
            

        logger.info(f'Dumping experiment to {out_dir}')
        os.makedirs(out_dir, exist_ok=True)

        for name, ext in to_load:

            # File path
            fp = path.join(out_dir, f'{name}.{ext}')

            # Saving function depending on file extension
            match ext:
                case 'npy': save_fun = np.save
                case 'npz': save_fun = lambda file, x: np.savez(file, **x)

            # Saving state
            match name:
                case 'labels' | 'scores_nat':  check_fn = lambda x : x.size
                case 'rf_maps':                check_fn = lambda x : len(x)
                case 'states':                 check_fn = lambda x : all([v.size for v in x.values()])
                case _:                        check_fn = lambda _ : True
                
            state = self.__getattribute__(name)
            
            if check_fn(state):
                logger.info(f"> Saving {name} history from {fp}")
                save_fun(fp, state)
            else:
                logger.warn(f'> Attempting to dump {name}, but empty')
        
        logger.info(f'')


class Experiment(ABC):
    '''
    This class implements the main pipeline of the Xdream experiment providing
    a granular implementation of the data flow.
    '''

    EXPERIMENT_TITLE = "Experiment"
    '''
    Experiment title. All experiment outputs will be
    saved in a sub-directory with its name.
    '''

    @classmethod
    def from_config(cls, conf : Dict[str, Any]) -> 'Experiment':
        
        experiment = cls._from_config(conf=conf)

        conf.pop('display_screens', None)
        
        experiment._set_param_configuration(param_config=conf)

        return experiment
    

    @classmethod
    @abstractmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'Experiment':
        pass
    
    def __init__(
        self,    
        generator:      Generator,
        subject:        InSilicoSubject,
        scorer:         Scorer,
        optimizer:      Optimizer,
        iteration:      int,
        logger:         Logger,
        mask_generator: MaskGenerator | None = None,
        data:           Dict[str, Any] = dict(),
        name:           str = 'experiment'
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
        :param iteration: Number of iteration to perform in the experiment.
        :type iteration: int
        :param logger: Logger instance to log information, warnings and errors,
                       to handle directory paths and to display screens.
        :type logger: Logger
        :param mask_generator: Function for mask generation. The boolean mask discriminates between synthetic
                               and natural images in the stimuli. A new mask is potentially generated at 
                               each iteration for a varying number of synthetic stimuli: the function
                               maps their number to a valid mask (i.e. as many True values as synthetic stimuli).
                               Defaults to None.
        :type mask_generator: MaskGenerator | None
        :param data: General purpose attribute to allow the inclusion of any additional type of information.
        :type data: Dict[str, Any]
        :type config: ExperimentConfig
        :param name: Name identifier for the experiment version, defaults to 'experiment'.
        :type name: str, optional
        '''
        
        # Experiment name
        self._name = name
        
        # Configuration attributes
        self._generator = generator
        self._subject   = subject
        self._scorer    = scorer
        self._optimizer = optimizer
        self._logger    = logger
        self._iteration = iteration
        
        # Defaults mask generator to a None mask, which is handled 
        # by the generator as a synthetic-only stimuli.
        self._mask_generator = cast(
            MaskGenerator,
            default(mask_generator, lambda x: np.array([], dtype=np.bool_))
        )

        # NOTE: `Data` input is not used in the default version, but
        #       it can exploited in subclasses to store additional information

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

    def __str__(self)  -> str: return f'{self.EXPERIMENT_TITLE}[{self._name}]'
    def __repr__(self) -> str: return str(self)

        
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
    
    @property
    def num_imgs(self) -> int:
        return self._optimizer.n_states
    
    # --- DATAFLOW METHODS ---
    
    # The following methods implements the main experiment data pipeline:
    #
    # Codes --Generator--> Stimuli --Subject--> State --Scorer--> Score --Optimizer--> Codes
    #
    # Along with component-specific data, a Message travels all along the pipeline. 
    # It allows for a generic data passing between components to communicate an information 
    # shared to entire system (e.g. the mask indicating synthetic and natural images).
    #
    # NOTE Experiment subclasses that intend to manipulate the data in the pipeline must
    #      override the specific function. The overridden method is in general advised not to
    #      redefine the default behavior but to redirect to it using the `super` keyboard.
    
    def _run_init(self, msg : Message) -> Tuple[Codes, Message]:
        # Codes initialization
        codes = self.optimizer.init()
        
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg
    
    def _codes_to_inputs(self, codes: Tuple[Codes, Message]) -> Tuple[Codes, Message]:
        '''
        This method is put here as a hook that can be overridden by an
        experiment in case some manipulation of the codes (e.g. reshaping)
        is needed to properly format them to suit the generator inputs.
        
        :param codes: Codes representing the Optimizer output
        :type codes: Codes
        :return: A new set of codes that suit the generator as inputs
        :rtype: Codes
        '''
        return codes

    def _codes_to_stimuli(self, data: Tuple[Codes, Message]) -> Tuple[Stimuli, Message]:
        '''
        The method uses the generator that maps input codes to visual stimuli.
        It uses the mask generator to generate the mask that interleaves
        synthetic and natural stimuli, and includes it in the Message.

        :param codes: Codes representing visual stimuli in the latent optimization space.
        :type codes: Codes
        :return: Generated visual stimuli and a Message
        :rtype: Tuple[Stimuli, Message]
        '''
        
        codes, msg = data
        
        # Generate the mask based on the number of codes 
        # produced by the optimizer.
        msg.mask = self._mask_generator(self.num_imgs)
        msg.masks_history.append(msg.mask)
        
        data = (codes, msg)
        
        return self.generator(data)
    
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[State, Message]:
        '''
        The method collects the Subject responses to a presented set of stimuli.

        :param data: Set of visual stimuli and a Message
        :type data: Tuple[Stimuli, Message]
        :return: Subject responses to the presented Stimuli and a Message.
        :rtype: Tuple[State, Message]
        '''
        
        states, msg = self.subject(data=data)
    
        return states, msg
    
    def _sbj_state_to_scr_state(self, sbj_state : Tuple[State, Message]) -> Tuple[State, Message]:
        '''
        
        '''
        
        return sbj_state
    
    def _scr_state_to_stm_score(self, data: Tuple[State, Message]) -> Tuple[Score, Message]:
        '''
        The method evaluate the SubjectResponse in light of a Scorer logic.

        :param data: Subject responses to visual stimuli and a Message
        :type data: Tuple[State, Message]
        :return: A score for each presented stimulus.
        :rtype: Tuple[Score, Message]
        '''
        
        scores, msg = self.scorer(data=data)
    
        msg.scores_gen_history.append(scores[ msg.mask])
        msg.scores_nat_history.append(scores[~msg.mask])
        
        return scores, msg
    
    def _stm_score_to_codes(self, data: Tuple[Score, Message]) -> Tuple[Codes, Message]:
        '''
        The method uses the scores for each stimulus to optimize the images
        in the latent coded space, resulting in a new set of codes.
        
        NOTE: The optimization only works with synthetic images so the optimizer
              is asked to filter out natural ones using mask information in the message. 
        
        NOTE: Depending on the specific Optimizer implementation the number of 
              codes at each iteration is possibly varying.

        :param data: Score associated to the score and a Message 
        :type data: Tuple[Score, Message]
        :return: New optimized set of codes.
        :rtype: Codes
        '''
        
        scores, msg = data
        
        # Filter scores only for synthetic images
        scores = scores[msg.mask]
        
        codes, msg = self.optimizer.step(data=(scores, msg))
    
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg
    
    # --- RUN ---
    
    # The general experiment pipeline is broken in different subfunctions
    # The granularity helps in the definition of a new experiment by
    # the modification of a specific component in the overall process.
    
    # NOTE: The overriding of the function is in general suggested not to redefine
    #       the default behavior but to redirect to it using `super` keyboard.
    
    def _init(self) -> Message:
        '''
        The method is called before running the actual experiment.
        The default version logs the attributes of the components.
        It is supposed to contain all preliminary operations such as initialization.
        '''

        # Create experiment directory
        self._logger.create_target_dir()

        # Save and log parameters
        if self._param_config:

            flat_dict = flatten_dict(self._param_config)
            
            # Log
            self._logger.info(f"")
            self._logger.info(mess=str(self))
            self._logger.info(f"Parameters:")

            max_key_len = max(len(key) for key in flat_dict.keys()) + 1 # for padding

            for k, v in flat_dict.items():
                k_ = f"{k}:"
                self._logger.info(f'{k_:<{max_key_len}}   {v}')

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
        
                
        # We generate an initial message containing the start time
        msg = Message(
            start_time = time.time(),
            rec_units  = self.subject.target,
            scr_units  = self.scorer.scoring_units,
        )
        
        return msg

        
    def _finish(self, msg : Message) -> Message:
        '''
        The method is called at the end of the actual experiment.
        The default version logs the experiment elapsed time.
        It is supposed to perform operation of experiment results.
        '''

        # Log total elapsed time
        str_time = stringfy_time(sec=msg.elapsed_time)
        self._logger.info(mess=f"Experiment finished successfully. Elapsed time: {str_time} s.")
        self._logger.info(mess="")

        # Dump
        state = ExperimentState.from_msg(msg)
        state.dump(
            out_dir=path.join(self.target_dir, 'state'),
            logger=self._logger
        )

        # NOTE: The method is also supposed to close logger screens
        #       However this is not implemented in the default version 
        #       because it may be possible to keep screen active for
        #       other purposes, so the logic is lead to subclasses.
        
        return msg
    
    def _progress_info(self, i: int, msg : Message) -> str:
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
    
    def _progress(self, i: int, msg : Message):
        '''
        The method is called at the end of any experiment iteration.
        The default version logs the default progress information.

        :param i: Current iteration number.
        :type i: int
        '''
        self._logger.info(self._progress_info(i=i, msg=msg))
        
    def _run(self, msg : Message) -> Message:
        '''
        The method implements the core of the experiment.
        It initializes random codes and iterates combining 
        the dataflow in the pipeline for the configured number 
        of iterations.
        '''

        self._logger.info("Running...")
        
        opt_codes, msg = self._run_init(msg)
        
        for i in range(self._iteration):
            inp_codes, msg = self._codes_to_inputs       ((opt_codes, msg))
            s_stimuli, msg = self._codes_to_stimuli      ((inp_codes, msg))
            sbj_state, msg = self._stimuli_to_sbj_state  ((s_stimuli, msg))
            scr_state, msg = self._sbj_state_to_scr_state((sbj_state, msg))
            stm_score, msg = self._scr_state_to_stm_score((scr_state, msg))
            opt_codes, msg = self._stm_score_to_codes    ((stm_score, msg))

            self._progress(i=i, msg=msg)
            
        msg.end_time = time.time()
        msg.elapsed_time = msg.start_time - msg.end_time
            
        return msg
    
    def run(self) -> Message:
        '''
        The method implements the experiment logic by combining
        `init()`, `run()` and `finish()` methods. 
        It computes the total elapsed time of the experiment that can
        be accessed in the `finish()` method.
        '''
        
        msg = self._init()
        
        msg = self._run(msg)
        
        msg = self._finish(msg)
        
        return msg

class MultiExperiment:
    '''
    Class for running multiple versions of the same experiment
    varying the set of hyperparameters through a configuration file.
    '''
    
    def __init__(
        self,
        experiment      : Type['Experiment'],
        experiment_conf : Dict[str, List[Any]], 
        default_conf  : Dict[str, Any],
    ) -> None:
        '''
        Create a list of experiments configurations by combining
        the provided experiment configuration with default ones.

        :param experiment: Experiment class for conducing multi-experiment run.
        :type experiment: Type['Experiment']
        :param experiment_conf: Configuration for the multi-experiment specifying a list
                                of values for a set of hyperparameters. All lists are
                                expected to have the same length.        
        :param default_conf: Default configuration file for all hyperparameters not 
                               explicitly provided in the experiment configuration.
        :type default_conf: Dict[str, Any]
        '''
        
        # Check that provided search configuration argument
        # share the same length as length defines the number
        # of experiments to run.
        lens = [len(v) for v in experiment_conf.values()]
        if len(set(lens)) > 1:
            err_msg = f'Provided search configuration files have conflicting '\
                      f'number of experiments runs: {lens}. Please provide '\
                      f'coherent number of experiment runs.'
            raise ValueError(err_msg)
        
        self._Exp = experiment
        
        # Convert the search configuration and combine with default values 
        keys, vals = experiment_conf.keys(), experiment_conf.values()

        self._search_config : List[Dict[str, Any]] = [
            overwrite_dict(
                default_conf, 
                {k : v for k, v in zip(keys, V)}
            )
            for V in zip(*vals)
        ]

        # Set the logger with experiment names check
        self._logger = self._set_logger()


    def __len__(self) -> int:
        return len(self._search_config)

    def __str__(self) -> str:
        return f'MultiExperiment[{self._name}]({len(self)} versions)'
    
    def __repr__(self) -> str: return str(self)

    def _set_logger(self) -> Logger:
        '''
        Set a multi-experiment level logger and checks name consistency between experiments
        '''

        # Check single experiment name
        exp_names = [conf['logger']['name'] for conf in self._search_config]
        if len(set(exp_names)) > 1:
            err_msg = f'Multirun expects a unique experiment name, but multiple found {set(exp_names)}'
            raise ValueError(err_msg)

        # Check multiple output directories
        out_dirs = [conf['logger']['out_dir'] for conf in self._search_config]
        if len(set(out_dirs)) > 1:
            err_msg = f'Multirun expects a unique output directory, but multiple found {set(out_dirs)}'
            raise ValueError(err_msg)

        # Set multi-experiment name
        self._name    = exp_names[0]

        return self._logger_type(
            conf = {
                'out_dir': out_dirs[0],
                'title'  : self._Exp.EXPERIMENT_TITLE,
                'name'   : self._name
            }
        )
        
    @property
    def _logger_type(self) -> Type[Logger]:
        '''
        Returns multi-experiment level logger type. 
        NOTE: Subclasses that want to change the logger type only need
              to override this property.
        '''
        return Logger


    @property
    def target_dir(self) -> str: return self._logger.target_dir
    

    def _get_display_screens(self) -> List[DisplayScreen]:
        '''
        Returns the list of screens used in the experiment.
        
        NOTE: This is made to make all experiments share and reuse the same
        screen and prevent screen instantiation overhead and memory leaks.

        NOTE: The method is not a property because it's pre-executed to simulate
              attribute behavior even if not explicitly called and we want to avoid
              the creation of the `DisplayScreen` with no usage. 
        '''

        # In the default version we have no screen
        return []

    def _init(self):
        '''
        The method is called before running all the experiments.
        The default version logs the number of experiments to run, set the shared screens
        across involved experiments and initialize an empty dictionary where to store multi-run results.
        It is supposed to contain all preliminary operations such as initialization.
        '''

        # Initial logging
        self._logger.info(mess=f'RUNNING MULTIPLE VERSIONS ({len(self)}) OF EXPERIMENT {self._name}')
        

        # NOTE: Handling screen turns fundamental in handling
        #       overhead and memory during multi-experiment run.
        #       For this reason in the case at least one experiment 
        #       has the `render` option enabled we instantiate shared
        #       screens among all this experiment that are attached
        #       to the configuration dictionary.

        # Add screens if at least one experiment has `render` flag on
        if any(conf['render'] for conf in self._search_config):

            self._main_screen = DisplayScreen.set_main_screen() # keep reference
            self._screens     = self._get_display_screens()

            # Add screens to configuration to experiments with `render` flag on
            for conf in self._search_config:
                if conf['render']:
                    conf['display_screens'] = self._screens

        # Data dictionary
        # NOTE: This is a general purpose data-structure where to save
        #       experiment results. It will be stored as a .pickle file.
        self._data: Dict[str, Any] = dict()

    def _progress(self, exp: Experiment, i: int, msg: Message):
        '''
        Method called after running a single experiment.
        In the default version it only logs the progress.

        :param exp: Instance of the run experiment.
        :type exp: Experiment
        :param i: _Iteration of the multi-run.
        :type i: int
        '''

        j = i+1
        self._logger.info(mess=f'EXPERIMENT {j} OF {len(self)} RUN SUCCESSFULLY.')

    def _finish(self):
        '''
        Method called after all experiments are run.
        It stores the results contained in the `data` dictionary if not empty.
        '''

        str_time = stringfy_time(sec=self._elapsed_time)
        self._logger.info(mess=f'ALL EXPERIMENT RUN SUCCESSFULLY. ELAPSED TIME: {str_time} s.')

        # Save multi-run experiment as a .pickle file
        if self._data:
            out_fp = path.join(self.target_dir, 'data.pickle')
            self._logger.info(mess=f'Saving multi-experiment data to {out_fp}')
            store_pickle(data=self._data, path=out_fp)

    def _run(self):
        '''
        Run the actual multi-run by executing all experiments in 
        the provided configurations.
        '''
        for i, conf in enumerate(self._search_config):
            
            self._logger.info(mess=f'RUNNING EXPERIMENT {i+1} OF {len(self)}.')
            
            exp = self._Exp.from_config(conf=conf)
            
            msg = exp.run()
            #return msg
            
            self._progress(exp=exp, i=i, msg= msg)

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
        #msg = self._run()
        #return msg
        
        end_time = time.time()
        
        self._elapsed_time = end_time - start_time
        
        self._finish()
    
