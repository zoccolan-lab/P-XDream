
from functools import partial
from os import path
import os
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image


from script.ClusteringOptimization.plotting import plot_scr
from zdream.clustering.ds import DSCluster, DSClusters
from zdream.experiment import Experiment, MultiExperiment
from zdream.generator import Generator, InverseAlexGenerator
from zdream.logger import Logger, LoguruLogger
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.dataset import MiniImageNet
from zdream.utils.model import Codes, DisplayScreen, MaskGenerator, Score, ScoringUnit, State, Stimuli, UnitsMapping, mask_generator_from_template
from zdream.utils.misc import concatenate_images, device
from zdream.utils.parsing import parse_boolean_string
from zdream.message import Message

# --- EXPERIMENT CLASS ---

class ClusteringOptimizationExperiment(Experiment):

    EXPERIMENT_TITLE = "ClusteringOptimization"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'

    @property
    def scorer(self)  -> ActivityScorer: return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> NetworkSubject:    return cast(NetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'ClusteringOptimizationExperiment':

        # Extract specific configurations
        gen_conf = conf['generator']
        msk_conf = conf['mask_generator']
        clu_conf = conf['clustering']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # Set random seed
        np.random.seed(conf['random_seed'])

        # --- CLUSTERING ---

        # Read clustering from file and extract the i-th cluster as specified by conf
        clusters: DSClusters = DSClusters.from_file(clu_conf['cluster_file'])
        cluster:  DSCluster  = clusters[clu_conf['cluster_idx']]

        # Choose random units depending on their random type
        tot_obj = clusters.obj_tot_count # tot number of objects
        clu_obj = len(cluster)           # cluster object count

        scoring_units: ScoringUnit
        match clu_conf['scr_type']:

            # 1) Cluster - use clusters units
            case 'cluster': 

                scoring_units = cluster.scoring_units

                # Get units mapping for the weighted sum
                units_mapping: UnitsMapping = partial(
                    cluster.units_mapping,
                    weighted=clu_conf['weighted_score']
                )

                units_reduction = 'sum'

            # 2) Random - use scattered random units with the same dimensionality of the cluster
            case 'random': 
                
                scoring_units = list(np.random.choice(
                    np.arange(0, tot_obj), 
                    size=clu_obj, 
                    replace=False
                ))

                units_mapping = lambda x: x
                units_reduction  = 'mean'
            
            # 3) Random adjacent - use continuos random units with the same dimensionality of the cluster
            case 'random_adj':

                start = np.random.randint(0, tot_obj - clu_obj + 1)
                scoring_units = list(range(start, start + clu_obj))

                units_mapping = lambda x: x
                units_reduction  = 'mean'
    
            # Default - raise an error
            case _:
                err_msg = f'Invalid `scr_type`: {clu_conf["scr_type"]}. '\
                           'Choose one between {cluster, random, random_adj}. '
                raise ValueError(err_msg)        

        # Extract layer idx clustering was performed with
        layer_idx = clu_conf['layer']

        # --- MASK GENERATOR ---

        template = parse_boolean_string(boolean_str=msk_conf['template'])
        mask_generator = mask_generator_from_template(template=template, shuffle=msk_conf['shuffle'])
        
        # --- GENERATOR ---

        # Dataloader
        use_nat = template.count(False) > 0

        if use_nat:
            dataset      = MiniImageNet(root=gen_conf['mini_inet'])
            dataloader   = DataLoader(dataset, batch_size=gen_conf['batch_size'], shuffle=True)

        # Instance
        generator = InverseAlexGenerator(
            root           = gen_conf['weights'],
            variant        = gen_conf['variant'],
            nat_img_loader = dataloader if use_nat else None
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = NetworkSubject(network_name=sbj_conf['net_name']).layer_info

        # Probe

        # NOTE: Since cluster idx, i.e. units where to score from refers to activation space
        #       we are required to record to all the layer for this property to hold
        #
        #       For the FUTURE in case this is computational demanding we can record only
        #       cluster neurons and score from all.
        
        layer_name = list(layer_info.keys())[layer_idx]
        probe = RecordingProbe(target = {layer_name: None}) # type: ignore

        # Subject with attached recording probe
        sbj_net = NetworkSubject(
            record_probe=probe,
            network_name=sbj_conf['net_name']
        )
        
        sbj_net.eval()

        # --- SCORER ---

        scorer = ActivityScorer(
            scoring_units={layer_name: scoring_units},
            units_reduction=units_reduction,
            layer_reduction=scr_conf['layer_reduction'],
            units_map=units_mapping
        )

        # --- OPTIMIZER ---

        optim = GeneticOptimizer(
            states_shape   = generator.input_dim,
            random_seed    =     conf['random_seed'],
            random_distr   = opt_conf['random_distr'],
            mutation_rate  = opt_conf['mutation_rate'],
            mutation_size  = opt_conf['mutation_size'],
            population_size= opt_conf['pop_size'],
            temperature    = opt_conf['temperature'],
            num_parents    = opt_conf['num_parents'],
            topk           = opt_conf['topk']
        )

        #  --- LOGGER --- 

        log_conf['title'] = ClusteringOptimizationExperiment.EXPERIMENT_TITLE
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

                # Add screen for natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(400, 400))
                    )

        # --- DATA ---
        data = {
            'weighted_score' : clu_conf['weighted_score'],
            'cluster_idx'    : clu_conf['cluster_idx'],
            'render'         : conf['render'],
            'use_nat'        : use_nat,
            'close_screen'   : conf.get('close_screen', False),
        }

        # Experiment configuration
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            iteration      = conf['iter'],
            mask_generator = mask_generator,
            data           = data, 
            name           = log_conf['name']
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
        self._render        = cast(bool, data['render'])
        self._close_screen  = cast(bool, data['close_screen'])
        self._use_nat       = cast(bool, data['use_nat'])
        self._weighted      = cast(bool, data['weighted_score'])
        self._cluster_idx   = cast( int, data['cluster_idx'])

    def _progress_info(self, i: int, msg : Message) -> str:

        # We add to progress information relative to best and average score

        # Synthetic
        stat_gen = msg.stats_gen
        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()

        # Natural (if used)
        if self._use_nat:
            stat_nat = msg.stats_nat
            best_nat = cast(NDArray, stat_nat['best_score']).mean()

        # Format strings
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'

        # Additional description
        if self._use_nat:
            best_nat_str = f'{best_nat:.1f}'
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str} | best nat: {best_nat_str}'
        else:
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'

        # Combine with default one
        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super}{desc}'

    def _init(self) -> Message:

        msg = super()._init()

        # Data structure to save best score and best image
        if self._use_nat:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
        
        return msg

    def _progress(self, i: int, msg : Message):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.solution
        best_synthetic, _ = self.generator(data=(best_code, Message(mask=np.array([True]))))
        best_synthetic_img = to_pil_image(best_synthetic[0])

        # Get best natural image
        if self._use_nat:
            best_natural = self._best_nat_img
            
        # Update screens
        if self._render:

            # Synthetic
            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )
            

            # Natural
            if self._use_nat:
                self._logger.update_screen(
                    screen_name=self.NAT_IMG_SCREEN,
                    image=to_pil_image(best_natural)
                )   

    def _finish(self, msg : Message):

        super()._finish(msg)

        # Close screens
        if self._close_screen:
            self._logger.close_all_screens()

        # 1. Save visual stimuli (synthetic and natural)

        img_dir = path.join(self.target_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        self._logger.info(mess=f"Saving images to {img_dir}")

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen, _ = self.generator(data=(msg.solution, Message(mask=np.array([True]))))
        best_gen = best_gen[0] # remove 1 batch size

        # We retrieve the stored best natural image
        if self._use_nat:
            best_nat = self._best_nat_img

        # Saving images
        for img, label in [
            (to_pil_image(best_gen), 'best synthetic'),
            (to_pil_image(best_nat), 'best natural'),
            (concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli'),
        ] if self._use_nat else [
            (to_pil_image(best_gen), 'best synthetic')
        ]:
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)
        
        self._logger.info(mess='')
        
        return msg
        

    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[State, Message]:

        # We save the last set of stimuli
        self._stimuli, _ = data

        return super()._stimuli_to_sbj_state(data)

    def _stm_score_to_codes(self, data: Tuple[Score, Message]) -> Tuple[Codes, Message]:

        sub_score, msg = data

        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_nat:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._stm_score_to_codes((sub_score, msg))
    

# --- MULTI-EXPERIMENT ---

class UnitsWeightingMultiExperiment(MultiExperiment):

    def __init__(
            self, 
            experiment:      Type['ClusteringOptimizationExperiment'], 
            experiment_conf: Dict[str, List[Any]], 
            default_conf:    Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)

        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True
    
    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [
            DisplayScreen(
                title=ClusteringOptimizationExperiment.GEN_IMG_SCREEN, 
                display_size=(400, 400)
            )
        ]

        # Add screen for natural images if at least one will use it
        if any(
            parse_boolean_string(conf['mask_generator']['template']).count(False) > 0 
            for conf in self._search_config
        ):
            screens.append(
                DisplayScreen(
                    title=ClusteringOptimizationExperiment.NAT_IMG_SCREEN, 
                    display_size=(400, 400)
                )
            )

        return screens
    
    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'       ] = 'Comparison between cluster-weighted an arithmetic score average'
        self._data['cluster_idx'] = list()  # Cluster idx in the clustering
        self._data['scores'     ] = list()  # Scores across generation
        self._data['weighted'   ] = list()  # Boolean flag if scoring was weighted or not


    def _progress(self, exp: ClusteringOptimizationExperiment, msg : Message, i: int):

        super()._progress(exp, i, msg = msg)

        self._data['scores']      .append(msg.scores_gen_history)
        self._data['cluster_idx'].append(exp._cluster_idx)
        self._data['weighted']   .append(exp._weighted)    

    def _finish(self):
        
        super()._finish()
        
        plot_scr(
            cluster_idx  = self._data['cluster_idx'],
            weighted     = self._data['weighted'],
            scores       = self._data['scores'],
            out_dir      = self.target_dir,
            logger       = self._logger
        )