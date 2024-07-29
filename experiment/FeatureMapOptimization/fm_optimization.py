
import os
from os import path
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from experiment.ClusterOptimization.plot import plot_activations, plot_cluster_units_beststimuli, plot_ds_weigthed_score
from experiment.utils.args import ExperimentArgParams
from experiment.utils.parsing import parse_boolean_string, parse_recording, parse_scoring
from experiment.utils.misc import BaseZdreamMultiExperiment, make_dir
from experiment.utils.settings import FILE_NAMES
from zdream.clustering.cluster import Clusters
from zdream.clustering.ds import DSClusters
from zdream.experiment import ZdreamExperiment
from zdream.generator import Generator, DeePSiMGenerator
from zdream.optimizer import CMAESOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, TorchNetworkSubject
from zdream.utils.dataset import MiniImageNet, NaturalStimuliLoader
from zdream.utils.io_ import read_json
from zdream.utils.logger import DisplayScreen, Logger, LoguruLogger
from zdream.utils.message import ZdreamMessage
from zdream.utils.misc import concatenate_images, device
from zdream.utils.parameters import ArgParams, ParamConfig
from zdream.utils.probe import RecordingProbe
from zdream.utils.types import Codes, Scores, ScoringUnit, States, Stimuli, UnitsMapping

# --- EXPERIMENT CLASS ---

class FeatureMapOptimizationExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "FeatureMapOptimization"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # Define specific components to activate their behavior    

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'FeatureMapOptimizationExperiment':
        
        # Feature map
        PARAM_fm_idx         = int  (conf[ExperimentArgParams.FeatureMapIdx     .value])
        PARAM_fm_dir         = str  (conf[ExperimentArgParams.FeatureMapDir     .value])
        PARAM_fm_seg_type    = str  (conf[ExperimentArgParams.FMSegmentationType.value])
        PARAM_fm_key         = str  (conf[ExperimentArgParams.FMKey             .value])

        # Clustering 
        PARAM_clu_dir        = str  (conf[ExperimentArgParams.ClusterDir       .value])
        PARAM_clu_algo       = str  (conf[ExperimentArgParams.ClusterAlgo      .value])
        PARAM_clu_idx        = int  (conf[ExperimentArgParams.ClusterIdx       .value])
        PARAM_clu_layer      = int  (conf[ExperimentArgParams.ClusterLayer     .value])
        
        # Generator
        PARAM_gen_weights    = str  (conf[ExperimentArgParams.GenWeights       .value])
        PARAM_gen_variant    = str  (conf[ExperimentArgParams.GenVariant       .value])
        
        # Natural image
        PARAM_dataset        = str  (conf[ExperimentArgParams.Dataset          .value])
        PARAM_batch_size     = int  (conf[ExperimentArgParams.BatchSize        .value])
        PARAM_template       = str  (conf[ExperimentArgParams.Template         .value])
        PARAM_shuffle        = bool (conf[ExperimentArgParams.Shuffle          .value])
        
        # Subject
        PARAM_net_name       = str  (conf[ExperimentArgParams.NetworkName      .value])
        
        # Scorer
        PARAM_layer_red      = str  (conf[ExperimentArgParams.LayerReduction.   value])
        PARAM_units_red      = str  (conf[ExperimentArgParams.UnitsReduction.   value])
        
        # Optimizer
        PARAM_pop_size       = int  (conf[ExperimentArgParams.PopulationSize   .value])
        PARAM_sigma_0        = float(conf[ExperimentArgParams.Sigma0           .value])
        
        # Logger
        PARAM_output_dir     = str  (conf[ArgParams.OutputDirectory            .value])
        PARAM_exp_name       = str  (conf[ArgParams.ExperimentName             .value])
        PARAM_rand_seed      = int  (conf[ArgParams.RandomSeed                 .value])
        
        # Experiment
        PARAM_iter           = int  (conf[ArgParams.NumIterations              .value])
        PARAM_rand_seed      = int  (conf[ArgParams.RandomSeed                 .value])
        PARAM_render         = bool (conf[ArgParams.Render                     .value])

        # Set numpy random seed
        np.random.seed(PARAM_rand_seed)
        
        # --- NATURAL IMAGE LOADER ---

        # Parse template and derive if to use natural images
        template = parse_boolean_string(boolean_str=PARAM_template)
        use_nat  = template.count(False) > 0
        
        # Create dataset and loader
        dataset  = MiniImageNet(root=PARAM_dataset)
        nat_img_loader = NaturalStimuliLoader(
            dataset=dataset,
            template=template,
            shuffle=PARAM_shuffle,
            batch_size=PARAM_batch_size,
        )
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root=PARAM_gen_weights,
            variant=PARAM_gen_variant # type: ignore - literal
        ).to(device)
        
        # --- FEATURE MAP ---
        
        match PARAM_fm_seg_type:
            
            case 'fm'  : 
                optim_file_name =  'fm_segmentation_optim.json'
                key = PARAM_fm_idx
            case 'clu' : 
                optim_file_name = 'clu_segmentation_optim.json'
                key = PARAM_clu_idx
            case _     : raise ValueError(f'Invalid segmentation type: {PARAM_fm_seg_type}. Choose from {{`fm` or `clu`}}')
        
        optim_dict = read_json(path.join(PARAM_fm_dir, optim_file_name))
        
        clu_algo_name = FILE_NAMES[PARAM_clu_algo].replace('Clusters.json', '')
        
        rec_units = (optim_dict[clu_algo_name][str(key)][PARAM_fm_key])
        
        # --- SUBJECT ---
        
        # Extract layer idx clustering was performed with
        layer_idx = PARAM_clu_layer

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=PARAM_net_name).layer_info

        # Probe
        rec_stringfy = " ".join([f"({' '.join([str(u) for u in rec_unit])})" for rec_unit in rec_units])
        
        rec_units = parse_recording(
            input_str = f'{layer_idx}=[{rec_stringfy}]',
            net_info=layer_info
        )
        
        layer_name = list(layer_info.keys())[layer_idx]         
        probe      = RecordingProbe(target = rec_units)

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name
        ).to(device)

        sbj_net.eval()
        

        # --- SCORER ---
        
        scr_units = parse_scoring(
            input_str = f'{layer_idx}=[]', 
            net_info  = layer_info,
            rec_info  = rec_units
        )

        scorer = ActivityScorer(
            scoring_units=scr_units,
            units_reduction=PARAM_units_red,
            layer_reduction=PARAM_layer_red
        )

        # --- OPTIMIZER ---
        
        optim = CMAESOptimizer(
            codes_shape  = generator.input_dim,
            rnd_seed     = PARAM_rand_seed,
            pop_size     = PARAM_pop_size,
            sigma0       = PARAM_sigma_0
        )

        #  --- LOGGER --- 

        conf[ArgParams.ExperimentTitle.value] = FeatureMapOptimizationExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
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
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=(800, 800))
                )

                # Add screen for natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(800, 800))
                    )

        # --- DATA ---
        
        data = {
            'render'         : PARAM_render,
            'use_nat'        : use_nat,
            'close_screen'   : conf.get(ArgParams.CloseScreen.value, False)
        }

        # Experiment configuration
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            iteration      = PARAM_iter,
            nat_img_loader = nat_img_loader,
            data           = data, 
            name           = PARAM_exp_name
        )

        return experiment

    def __init__(
        self,    
        generator      : Generator,
        subject        : InSilicoSubject,
        scorer         : Scorer,
        optimizer      : Optimizer,
        iteration      : int,
        logger         : Logger,
        nat_img_loader : NaturalStimuliLoader,
        data           : Dict[str, Any] = dict(),
        name           : str = 'clustering-optimization'
    ) -> None:

        super().__init__(
            generator      = generator,
            subject        = subject,
            scorer         = scorer,
            optimizer      = optimizer,
            iteration      = iteration,
            logger         = logger,
            nat_img_loader = nat_img_loader,
            data           = data,
            name           = name,
        )

        # Extract from Data
        self._render         = cast(bool,                   data['render'])
        self._close_screen   = cast(bool,                   data['close_screen'])
        self._use_nat        = cast(bool,                   data['use_nat'])

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:

        # Synthetic
        stat_gen = msg.stats_gen
        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()

        # Format strings
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        
        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'
        
        # Natural (if used)
        if self._use_nat:
            stat_nat = msg.stats_nat
            best_nat = cast(NDArray, stat_nat['best_score']).mean()
            best_nat_str = f'{best_nat:.1f}'
            desc = f'{desc} | best nat: {best_nat_str}'
        
        # Combine with default one
        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super}{desc}'

    def _init(self) -> ZdreamMessage:

        msg = super()._init()

        # Data structure to save best score and best image
        if self._use_nat:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
        
        return msg

    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)
        
        # Update screens
        if self._render:
            
            # Get best stimuli
            best_code          = msg.best_code
            best_synthetic     = self.generator(codes=best_code)
            best_synthetic_img = to_pil_image(best_synthetic[0])

            # Synthetic
            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )
        
            # Natural
            if self._use_nat:
                self._logger.update_screen(
                    screen_name=self.NAT_IMG_SCREEN,
                    image=to_pil_image(self._best_nat_img)
                )   

    def _finish(self, msg : ZdreamMessage):

        super()._finish(msg)

        # Close screens
        if self._close_screen: self._logger.close_all_screens()
            
        # 1) SAVING BEST STIMULI

        # Save visual stimuli (synthetic and natural)
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.best_code)

        # We retrieve the stored best natural image
        if self._use_nat: best_nat = self._best_nat_img

        # Saving images
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen[0]), 'best synthetic')]
        
        if self._use_nat:
            to_save.extend([
                (to_pil_image(best_nat), 'best natural'),
                (concatenate_images(img_list=[best_gen[0], best_nat]), 'best stimuli'),
            ])
        
        for img, name in to_save:
            
            out_fp = path.join(img_dir, f'{name.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {name} image to {out_fp}')
            img.save(out_fp)
        
        self._logger.info(mess='')
        
        return msg
    
    # --- PIPELINE ---
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:

        # We save the last set of stimuli
        self._stimuli, _ = data

        return super()._stimuli_to_states(data)

    def _scores_to_codes(self, data: Tuple[Scores, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:

        sub_score, msg = data

        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_nat:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._scores_to_codes((sub_score, msg))


# --- MULTI-EXPERIMENT ---

class FeatureMapSegmentsMultiExperiment(BaseZdreamMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc']  = 'Feature map optimization'
        
        # Dictionary: key: code / activation
        self._data['key_superstimulus'] = defaultdict(lambda: defaultdict(lambda: {'fitness': 0., 'code': None}))


    def _progress(
        self, 
        exp  : FeatureMapOptimizationExperiment,
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
                
        fm_seg_type = conf[ExperimentArgParams.FMSegmentationType.value] 
        
        match fm_seg_type:
            
            case 'fm'  : key = conf[ExperimentArgParams.FeatureMapIdx.value]
            case 'clu' : key = conf[ExperimentArgParams.ClusterIdx   .value]
            case  _    : raise ValueError(f'Invalid segment type {fm_seg_type}. Must be either {{ `fm`, `clu`}} ')
        
        fm_key   = conf[ExperimentArgParams.FMKey.value]
        code     = msg.best_code[0]
        fitness  = float(msg.stats_gen['best_score'])
        
        if key not in self._data['key_superstimulus']:
            
            self._data['key_superstimulus'][key][fm_key]['fitness'] = (fitness)
            self._data['key_superstimulus'][key][fm_key]['code'   ] = code
        
        if fitness > self._data['key_superstimulus'][key][fm_key]['fitness']:
            
            self._data['key_superstimulus'][key][fm_key]['fitness'] = (fitness)
            self._data['key_superstimulus'][key][fm_key]['code'   ] = code

    def _finish(self):
        
        # Dictionary redefinition with no lambdas
        # SEE: https://stackoverflow.com/questions/72339545/attributeerror-cant-pickle-local-object-locals-lambda
        
        self._data['key_superstimulus'] = {
            idx: {k: info for k, info in fm_info.items()} 
            for idx, fm_info in self._data['key_superstimulus'].items()
        }
        
        super()._finish()
        
        # NOTE: The plotting function requires fitness cardinality normalization so 
        #       The plot is supposed to be performed not from the multi-experiment
