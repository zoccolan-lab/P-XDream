
from os import path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from torchvision.transforms.functional import to_pil_image

from PIL import Image

from experiment.OptimizerTuning.plot import plot_hyperparam, plot_optim_type_comparison
from experiment.utils.args import ExperimentArgParams
from experiment.utils.misc import BaseZdreamMultiExperiment, make_dir
from pxdream.experiment import ZdreamExperiment
from pxdream.generator import Generator, DeePSiMGenerator
from pxdream.utils.logger import DisplayScreen, Logger, LoguruLogger
from pxdream.optimizer import CMAESOptimizer, GeneticOptimizer, Optimizer
from pxdream.scorer import ActivityScorer, Scorer
from pxdream.subject import InSilicoSubject, TorchNetworkSubject
from pxdream.utils.parameters import ArgParam, ArgParams, ParamConfig, Parameter
from pxdream.utils.probe import RecordingProbe
from pxdream.utils.misc import device
from experiment.utils.parsing import parse_recording, parse_scoring
from pxdream.utils.message import ZdreamMessage

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
    def _from_config(cls, conf : ParamConfig) -> 'OptimizationTuningExperiment':

        # Generator
        PARAM_gen_weights  = str  (conf[ExperimentArgParams.GenWeights       .value])
        PARAM_gen_variant  = str  (conf[ExperimentArgParams.GenVariant       .value])
        PARAM_net_name     = str  (conf[ExperimentArgParams.NetworkName      .value])

        # Subject
        PARAM_rec_layers   = str  (conf[ExperimentArgParams.RecordingLayers  .value])
        
        # Scorer
        PARAM_scr_layers   = str  (conf[ExperimentArgParams.ScoringLayers    .value])
        PARAM_units_red    = str  (conf[ExperimentArgParams.UnitsReduction   .value])
        PARAM_layer_red    = str  (conf[ExperimentArgParams.LayerReduction   .value])
    
        # Optimizer
        PARAM_optim_type   = str  (conf[ExperimentArgParams.OptimType        .value])
        PARAM_rand_distr   = str  (conf[ExperimentArgParams.RandomDistr      .value])
        PARAM_rand_scale   = float(conf[ExperimentArgParams.RandomScale      .value])
        PARAM_pop_size     = int  (conf[ExperimentArgParams.PopulationSize   .value])
        PARAM_mut_size     = float(conf[ExperimentArgParams.MutationSize     .value])
        PARAM_mut_rate     = float(conf[ExperimentArgParams.MutationRate     .value])
        PARAM_num_parents  = int  (conf[ExperimentArgParams.NumParents       .value])
        PARAM_allow_clones = bool (conf[ExperimentArgParams.AllowClones      .value])
        PARAM_topk         = int  (conf[ExperimentArgParams.TopK             .value])
        PARAM_temp         = float(conf[ExperimentArgParams.Temperature      .value])
        PARAM_temp_factor  = float(conf[ExperimentArgParams.TemperatureFactor.value])
        PARAM_sigma0       = float(conf[ExperimentArgParams.Sigma0           .value])
    
        # Logger
        PARAM_exp_name    = str  (conf[ArgParams          .ExperimentName   .value])
        PARAM_exp_version = str  (conf[ArgParams          .ExperimentVersion.value])
        PARAM_out_dir     = str  (conf[ArgParams          .OutputDirectory  .value])
    
        # Globals
        PARAM_iter        = int  (conf[ArgParams          .NumIterations    .value])
        PARAM_rand_seed   = int  (conf[ArgParams          .RandomSeed       .value])
        PARAM_render      = bool (conf[ArgParams          .Render           .value])

        # Set numpy random seed
        np.random.seed(PARAM_rand_seed)

        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root           = PARAM_gen_weights,
            variant        = PARAM_gen_variant  # type: ignore - literal
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=PARAM_net_name).layer_info

        # Probe
        record_target = parse_recording(input_str=PARAM_rec_layers, net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name
        )
        
        sbj_net.eval()

        # --- SCORER ---

        # Target neurons
        scoring_units = parse_scoring(
            input_str=PARAM_scr_layers, 
            net_info=layer_info,
            rec_info=record_target
        )

        scorer = ActivityScorer(
            scoring_units=scoring_units,
            units_reduction=PARAM_units_red,
            layer_reduction=PARAM_layer_red,
        )

        # --- OPTIMIZER ---
        
        opt_type = PARAM_optim_type
        match opt_type:
            
            case 'genetic':
        
                optim =  GeneticOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     = PARAM_rand_seed,
                    rnd_distr    = PARAM_rand_distr, # type: ignore - literal
                    rnd_scale    = PARAM_rand_scale,
                    pop_size     = PARAM_pop_size,
                    mut_size     = PARAM_mut_size,
                    mut_rate     = PARAM_mut_rate,
                    n_parents    = PARAM_num_parents,
                    allow_clones = PARAM_allow_clones,
                    topk         = PARAM_topk,
                    temp         = PARAM_temp,
                    temp_factor  = PARAM_temp_factor,
                )
                
            case 'cmaes': 
                
                optim = CMAESOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     = PARAM_rand_seed,
                    pop_size     = PARAM_pop_size,
                    sigma0       = PARAM_sigma0
                )
                
            case _: 
                
                raise ValueError(f'Invalid optimizer: {opt_type}')

        #  --- LOGGER --- 

        conf[ArgParams.ExperimentTitle.value] = OptimizationTuningExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger.from_conf(conf=conf)
        
        # In the case render option is enabled we add display screens
        if PARAM_render:

            # In the case of multi-experiment run, the shared screens
            # are set in `str(Args.DisplayScreens)` entry
            if ArgParams.DisplayScreens.value in conf:
                for screen in conf[ArgParams.DisplayScreens.value]:  # type: ignore
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
            'optim_type'   : PARAM_optim_type,
            'render'       : PARAM_render,
            'close_screen' : conf.get(ArgParams.CloseScreen.value, False),
        }

        # Experiment configuration
        experiment = cls(
            generator = generator,
            scorer    = scorer,
            optimizer = optim,
            subject   = sbj_net,
            logger    = logger,
            iteration = PARAM_iter,
            data      = data, 
            name      = PARAM_exp_name
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
            best_code = msg.best_code
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
        best_gen = self.generator(codes=msg.best_code)

        # Saving images
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen[0]), 'best synthetic')]
        
        for img, name in to_save:
            
            out_fp = path.join(img_dir, f'{name.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {name} image to {out_fp}')
            img.save(out_fp)
        
        self._logger.info(msg='')
        
        return msg
        

# --- MULTI-EXPERIMENT ---

class OptimizerComparisonMultiExperiment(BaseZdreamMultiExperiment):
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'      ] = 'Comparison between scores trend for the two optimizers'
        self._data['optim_type'] = list()  # Type of optimizer used
        self._data['scores'    ] = list()  # Scores across generation


    def _progress(
        self, 
        exp  : OptimizationTuningExperiment, 
        conf : ParamConfig,
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

class HyperparameterTuningMultiExperiment(BaseZdreamMultiExperiment):
    
    def __init__(
        self,
        experiment: type[OptimizationTuningExperiment], 
        experiment_conf: Dict[ArgParam, List[Parameter]],
        default_conf: ParamConfig
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
        conf : ParamConfig,
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
        
