
from functools import partial
from os import path
import os
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from PIL import Image


from script.ClusteringOptimization.plotting import plot_scr, plot_weighted
from zdream.clustering.ds import DSCluster, DSClusters
from zdream.experiment import ZdreamExperiment, MultiExperiment
from zdream.generator import Generator, InverseAlexGenerator
from zdream.logger import Logger, LoguruLogger
from zdream.optimizer import CMAESOptimizer, GeneticOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.dataset import MiniImageNet
from zdream.utils.model import Codes, DisplayScreen, MaskGenerator, Score, ScoringUnit, State, Stimuli, UnitsMapping, mask_generator_from_template
from zdream.utils.misc import concatenate_images, device
from zdream.utils.parsing import parse_boolean_string, parse_recording, parse_scoring
from zdream.message import ZdreamMessage

# --- EXPERIMENT CLASS ---

class OptimizationTuningExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "OptimizationTuning"

    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # Define specific components to activate their behavior    

    @property
    def scorer(self)  -> ActivityScorer: return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> NetworkSubject: return cast(NetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'OptimizationTuningExperiment':

        # Extract components configurations
        gen_conf = conf['generator']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # Set numpy random seed
        np.random.seed(conf['random_seed'])

        # --- GENERATOR ---

        generator = InverseAlexGenerator(
            root           = gen_conf['weights'],
            variant        = gen_conf['variant']
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = NetworkSubject(network_name=sbj_conf['net_name']).layer_info

        # Probe
        record_target = parse_recording(input_str=sbj_conf['rec_layers'], net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = NetworkSubject(
            record_probe=probe,
            network_name=sbj_conf['net_name']
        )
        
        sbj_net.eval()

        # --- SCORER ---

        # Target neurons
        scoring_units = parse_scoring(
            input_str=scr_conf['scr_layers'], 
            net_info=layer_info,
            rec_info=record_target
        )

        scorer = ActivityScorer(
            scoring_units=scoring_units,
            units_reduction=scr_conf['units_reduction'],
            layer_reduction=scr_conf['layer_reduction'],
        )

        # --- OPTIMIZER ---
        
        match opt_conf['optimizer_type']:
            
            case 'genetic':
        
                optim =  GeneticOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     =     conf['random_seed'],
                    rnd_distr    = opt_conf['random_distr'],
                    rnd_scale    = opt_conf['random_scale'],
                    pop_size     = opt_conf['pop_size'],
                    mut_size     = opt_conf['mutation_size'],
                    mut_rate     = opt_conf['mutation_rate'],
                    n_parents    = opt_conf['num_parents'],
                    allow_clones = opt_conf['allow_clones'],
                    topk         = opt_conf['topk'],
                    temp         = opt_conf['temperature'],
                    temp_factor  = opt_conf['temperature_factor']
                )
                
            case 'cmaes': 
                
                optim = CMAESOptimizer(
                    codes_shape  = generator.input_dim,
                    rnd_seed     =     conf['random_seed'],
                    rnd_distr    = opt_conf['random_distr'],
                    rnd_scale    = opt_conf['random_scale'],
                    pop_size     = opt_conf['pop_size'],
                    sigma0       = opt_conf['sigma0']
                )
                
            case _: 
                
                raise ValueError(f'Invalid optimizer: {opt_conf["optim_type"]}')

        #  --- LOGGER --- 

        log_conf['title'] = OptimizationTuningExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(conf=log_conf)
        
        # In the case render option is enabled we add display screens
        if conf['render']:

            # In the case of multi-experiment run, the shared screens
            # are set in `display_screens` entry
            if 'display_screens' in conf:
                for screen in conf['display_screens']:
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
        mask_generator: MaskGenerator | None = None,
        data:           Dict[str, Any] = dict(),
        name:           str = 'clustering-optimization'
    ) -> None:

        super().__init__(
            generator      = generator,
            subject        = subject,
            scorer         = scorer,
            optimizer      = optimizer,
            iteration      = iteration,
            logger         = logger,
            mask_generator = mask_generator,
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

        # Get best stimuli
        best_code = msg.solution
        best_synthetic, _ = self.generator(data=(best_code, ZdreamMessage(mask=np.array([True]))))
        best_synthetic_img = to_pil_image(best_synthetic[0])
            
        # Update screen
        if self._render:

            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )

    def _finish(self, msg : ZdreamMessage):

        super()._finish(msg)

        # Close screens
        if self._close_screen:
            self._logger.close_all_screens()
            
        # 1) SAVING BEST STIMULI

        # Save visual stimuli (synthetic and natural)
        img_dir = path.join(self.target_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        self._logger.info(mess=f"Saving images to {img_dir}")

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen, _ = self.generator(data=(msg.solution, ZdreamMessage(mask=np.array([True]))))

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


    def _progress(self, exp: OptimizationTuningExperiment, msg : ZdreamMessage, i: int):

        super()._progress(exp, i, msg = msg)

        self._data['optim_type'].append(exp._optim_type)
        self._data['scores']    .append(msg.scores_gen_history)   

    def _finish(self):
        
        super()._finish()
        
        # TODO PLOT

class HyperparameterTuningMultiExperiment(_OptimizerTuningMultiExperiment):
    
    def __init__(
        self,
        experiment: type[OptimizationTuningExperiment], 
        experiment_conf: Dict[str, List[Any]],
        default_conf: Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)
        
        self.hyperparameter = ''
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'      ] = 'Comparison of same optimization optimizing a different hyperparameter'
        self._data['hyperparam'] = list()  # Cluster idx in the clustering
        self._data['scores'    ] = list()  # Scores across generation

    def _progress(self, exp: OptimizationTuningExperiment, msg : ZdreamMessage, i: int):

        super()._progress(exp, i, msg = msg)

        self._data['hyperparam'].append(exp.optimizer.__getattribute__(f'_{self.hyperparameter}'))
        self._data['scores']    .append(msg.scores_gen_history)   

    def _finish(self):
        
        super()._finish()
        
        # TODO PLOT