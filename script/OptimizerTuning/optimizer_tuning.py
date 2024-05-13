
from os import path
import os
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
from numpy.typing import NDArray
from torchvision.transforms.functional import to_pil_image

from PIL import Image

from script.OptimizerTuning.plotting import plot_hyperparam, plot_optim_type_comparison
from script.utils.cmdline_args import Args
from script.utils.misc import make_dir
from zdream.experiment import ZdreamExperiment, MultiExperiment
from zdream.generator import Generator, DeePSiMGenerator
from zdream.utils.logger import DisplayScreen, Logger, LoguruLogger
from zdream.optimizer import CMAESOptimizer, GeneticOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, TorchNetworkSubject
from zdream.utils.probe import RecordingProbe
from zdream.utils.types import MaskGenerator
from zdream.utils.misc import device
from script.utils.parsing import parse_recording, parse_scoring
from zdream.utils.message import ZdreamMessage

# --- EXPERIMENT CLASS ---

class OptimizationTuningExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "OptimizationTuning"
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # Define specific components to activate their behavior    

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer,      self._scorer) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'OptimizationTuningExperiment':

        # Extract components configurations
        gen_conf = conf['generator']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # Set numpy random seed
        np.random.seed(conf[str(Args.RandomSeed)])

        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root           = gen_conf[str(Args.GenWeights)],
            variant        = gen_conf[str(Args.GenVariant)],
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=sbj_conf[str(Args.NetworkName)]).layer_info

        # Probe
        record_target = parse_recording(input_str=sbj_conf[str(Args.RecordingLayers)], net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=sbj_conf[str(Args.NetworkName)]
        )
        
        sbj_net.eval()

        # --- SCORER ---

        # Target neurons
        scoring_units = parse_scoring(
            input_str=scr_conf[str(Args.ScoringLayers)], 
            net_info=layer_info,
            rec_info=record_target
        )

        scorer = ActivityScorer(
            scoring_units=scoring_units,
            units_reduction=scr_conf[str(Args.UnitsReduction)],
            layer_reduction=scr_conf[str(Args.LayerReduction)],
        )

        # --- OPTIMIZER ---
        
        opt_type = opt_conf[str(Args.OptimType)]
        match opt_type:
            
            case 'genetic':
        
                optim =  GeneticOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     =     conf[str(Args.RandomSeed)],
                    rnd_distr    = opt_conf[str(Args.RandomDistr)],
                    rnd_scale    = opt_conf[str(Args.RandomScale)],
                    pop_size     = opt_conf[str(Args.PopulationSize)],
                    mut_size     = opt_conf[str(Args.MutationSize)],
                    mut_rate     = opt_conf[str(Args.MutationRate)],
                    n_parents    = opt_conf[str(Args.NumParents)],
                    allow_clones = opt_conf[str(Args.AllowClones)],
                    topk         = opt_conf[str(Args.TopK)],
                    temp         = opt_conf[str(Args.Temperature)],
                    temp_factor  = opt_conf[str(Args.TemperatureFactor)]
                )
                
            case 'cmaes': 
                
                optim = CMAESOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     =     conf[str(Args.RandomSeed)],
                    rnd_distr    = opt_conf[str(Args.RandomDistr)],
                    rnd_scale    = opt_conf[str(Args.RandomScale)],
                    pop_size     = opt_conf[str(Args.PopulationSize)],
                    sigma0       = opt_conf[str(Args.Sigma0)],
                )
                
            case _: 
                
                raise ValueError(f'Invalid optimizer: {opt_type}')

        #  --- LOGGER --- 

        log_conf[str(Args.ExperimentTitle)] = OptimizationTuningExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(path=log_conf)
        
        # In the case render option is enabled we add display screens
        if conf[str(Args.Render)]:

            # In the case of multi-experiment run, the shared screens
            # are set in `str(Args.DisplayScreens)` entry
            if str(Args.DisplayScreens) in conf:
                for screen in conf[str(Args.DisplayScreens)]:
                    logger.add_screen(screen=screen)

            # If the key is not set it is the case of a single experiment
            # and we create screens instance
            else:

                # Add screen for synthetic images
                logger.add_screen(
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=(400, 400))
                )
        
        # --- DATA ---
        
        data = {
            'optim_type'   : opt_conf['optimizer_type'],
            'render'       : conf['render'],
            'close_screen' : conf.get('close_screen', False),
        }

        # Experiment configuration
        experiment = cls(
            generator = generator,
            scorer    = scorer,
            optimizer = optim,
            subject   = sbj_net,
            logger    = logger,
            iteration = conf['iter'],
            data      = data, 
            name      = log_conf['name']
        )

        return experiment

    def __init__(
        self,    
        generator:      Generator,
        subject:        InSilicoSubject,
        scorer:         Scorer,
        optimizer:      Optimizer,
        iteration:      int,
        logger:         Logger,
        data:           Dict[str, Any] = dict(),
        name:           str = 'optmizer-tuning'
    ) -> None:

        super().__init__(
            generator      = generator,
            subject        = subject,
            scorer         = scorer,
            optimizer      = optimizer,
            iteration      = iteration,
            logger         = logger,
            data           = data,
            name           = name,
        )

        # Extract from Data
        self._optim_type    = cast(str,  data['optim_type'])
        self._close_screen  = cast(bool, data['close_screen'])
        self._render        = cast(bool, data['render'])

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:

        # We add to progress information relative to best and average score

        # Synthetic
        stat_gen = msg.stats_gen
        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()

        # Format strings
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'

        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'

        # Combine with default one
        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super}{desc}'


    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)

            
        # Update screen
        if self._render:
            
            # Get best stimuli
            best_code = msg.solution
            best_synthetic = self.generator(codes=best_code)
            best_synthetic_img = to_pil_image(best_synthetic[0])

            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )

    def _finish(self, msg : ZdreamMessage):

        super()._finish(msg)

        # Close screens
        if self._close_screen:
            self._logger.close_all_screens()

        # Save visual stimuli (synthetic and natural)
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.solution)

        # Saving images
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen[0]), 'best synthetic')]
        
        for img, name in to_save:
            
            out_fp = path.join(img_dir, f'{name.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {name} image to {out_fp}')
            img.save(out_fp)
        
        self._logger.info(mess='')
        
        return msg
        

# --- MULTI-EXPERIMENT ---

class _OptimizerTuningMultiExperiment(MultiExperiment):

    def __init__(
        self, 
        experiment:      Type['OptimizationTuningExperiment'], 
        experiment_conf: Dict[str, List[Any]], 
        default_conf:    Dict[str, Any],
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)

        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True
    
    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        return [
            DisplayScreen(
                title=OptimizationTuningExperiment.GEN_IMG_SCREEN, 
                display_size=(400, 400)
            )
        ]
    
    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger
    

class OptimizerComparisonMultiExperiment(_OptimizerTuningMultiExperiment):
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'      ] = 'Comparison between scores trend for the two optimizers'
        self._data['optim_type'] = list()  # Type of optimizer used
        self._data['scores'    ] = list()  # Scores across generation


    def _progress(
        self, 
        exp  : OptimizationTuningExperiment, 
        conf : Dict[str, Any],
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)

        self._data['optim_type'].append(exp._optim_type)
        self._data['scores']    .append(msg.scores_gen_history)   

    def _finish(self):
        
        super()._finish()
        
        plot_optim_type_comparison(
            opt_types = self._data['optim_type'],
            scores    = self._data['scores'],
            out_dir   = self.target_dir,
            logger    = self._logger
        )

class HyperparameterTuningMultiExperiment(_OptimizerTuningMultiExperiment):
    
    def __init__(
        self,
        experiment: type[OptimizationTuningExperiment], 
        experiment_conf: Dict[str, List[Any]],
        default_conf: Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)
        
        self.hyperparameter = ''
        self.score_title    = ''
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'      ] = 'Comparison of same optimization optimizing a different hyperparameter'
        self._data['hyperparam'] = self.hyperparameter
        self._data['values'    ] = list()  # Cluster idx in the clustering
        self._data['scores'    ] = list()  # Scores across generation

    def _progress(
        self, 
        exp  : OptimizationTuningExperiment, 
        conf : Dict[str, Any],
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)

        self._data['values'].append(exp.optimizer.__getattribute__(f'_{self.hyperparameter}'))
        self._data['scores'].append(msg.scores_gen_history)   

    def _finish(self):
        
        super()._finish()
        
        plot_hyperparam(
            hyperparam = self._data['hyperparam'],
            values     = self._data['values'],
            scores     = self._data['scores'],
            out_dir    = self.target_dir,
            logger     = self._logger
        )
        
