
import os
from os import path
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from experiment.ClusterOptimization.plotting import plot_activations, plot_cluster_best_stimuli, plot_cluster_target, plot_scr, plot_subsetting_optimization, plot_weighted
from experiment.utils.cmdline_args import Args
from experiment.utils.parsing import parse_boolean_string
from experiment.utils.misc import make_dir
from experiment.utils.settings import FILE_NAMES
from zdream.clustering.cluster import Clusters
from zdream.clustering.ds import DSClusters
from zdream.experiment import ZdreamExperiment, MultiExperiment
from zdream.generator import Generator, DeePSiMGenerator
from zdream.optimizer import CMAESOptimizer, GeneticOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, TorchNetworkSubject
from zdream.utils.dataset import MiniImageNet, NaturalStimuliLoader
from zdream.utils.logger import DisplayScreen, Logger, LoguruLogger
from zdream.utils.message import ZdreamMessage
from zdream.utils.misc import concatenate_images, device
from zdream.utils.probe import RecordingProbe
from zdream.utils.types import Codes, Scores, ScoringUnit, States, Stimuli, UnitsMapping

# --- EXPERIMENT CLASS ---

class ClusteringOptimizationExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "ClusteringOptimization"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # Define specific components to activate their behavior    

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'ClusteringOptimizationExperiment':

        # Extract components configurations
        gen_conf = conf['generator']
        nsl_conf = conf['natural_stimuli_loader']
        clu_conf = conf['clustering']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # Set numpy random seed
        np.random.seed(conf[str(Args.RandomSeed)])
        
        # --- NATURAL IMAGE LOADER ---

        # Parse template and derive if to use natural images
        template = parse_boolean_string(boolean_str=nsl_conf[str(Args.Template)])
        use_nat  = template.count(False) > 0
        
        # Create dataset and loader
        dataset  = MiniImageNet(root=nsl_conf[str(Args.Dataset)])
        nat_img_loader = NaturalStimuliLoader(
            dataset=dataset,
            template=template,
            shuffle=nsl_conf[str(Args.Shuffle)]
        )
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root=gen_conf[str(Args.GenWeights)],
            variant=gen_conf[str(Args.GenVariant)]
        ).to(device)
        
        # --- SUBJECT ---
        
        # Extract layer idx clustering was performed with
        layer_idx = clu_conf[str(Args.ClusterLayer)]

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=sbj_conf[str(Args.NetworkName)]).layer_info

        # Probe

        # NOTE: Since cluster idx, i.e. units where to score from refers to activation space
        #       we are required to record to all the layer for this property to hold
        #
        #       For the FUTURE in case this is computational demanding we can record only
        #       cluster neurons and score from all.
        
        layer_name = list(layer_info.keys())[layer_idx]
        probe = RecordingProbe(target = {layer_name: None}) # type: ignore

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=sbj_conf[str(Args.NetworkName)]
        ).to(device)

        sbj_net.eval()

        # --- CLUSTERING ---

        # Read clustering from file and extract the i-th cluster as specified by conf
        clu_type = clu_conf[str(Args.ClusterAlgo)]
        clu_fp   = path.join(clu_conf[str(Args.ClusterDir)], FILE_NAMES[clu_type])
        
        match clu_type:
            
            case 'ds':
                
                ds = True
                clusters: Clusters | DSClusters = DSClusters.from_file(fp=clu_fp)
                
            case 'gmm' | 'nc' | 'adj' | 'rand':
                
                ds = False
                clusters: Clusters | DSClusters = Clusters.from_file(fp=clu_fp)
            
            case _:
                
                raise ValueError(f'Invalid clustering type: {clu_type}. Choose one between `ds`, `gmm`, `nc`', '`adj`, `rand`')
        
        cluster = clusters[clu_conf[str(Args.ClusterIdx)]]

        # Choose random units depending on their random type
        tot_obj = clusters.obj_tot_count # tot number of objects
        clu_obj = len(cluster)           # cluster object count
        
        # Dictionary indicating groups of activations to store
        # mapping key name to neurons indexes
        activations_idx: Dict[str, ScoringUnit] = {}
        
        # Default unit reduction as mean
        units_mapping   = lambda x: x
        units_reduction = 'mean'
        
        # Retrieve all units idx in the layer
        layer_shape     = list(layer_info.values())[layer_idx]
        tot_layer_units = np.prod(layer_shape)
        layer_idx       = list(range(tot_layer_units))

        scr_type = clu_conf[str(Args.ScoringType)]
        
        scoring_units: ScoringUnit
        
        match scr_type:

            # 1) Cluster - use clusters units
            case 'cluster': 

                scoring_units = cluster.scoring_units

                # Get units mapping for the weighted sum
                if ds: 
                
                    units_mapping: UnitsMapping = partial(
                        cluster.units_mapping,
                        weighted=clu_conf[str(Args.WeightedScore)]
                    )
                    
                else:
                    
                    units_mapping: UnitsMapping = partial(
                        cluster.units_mapping
                    )

                units_reduction = 'sum'
                
                # Take at random other units outside the cluster with the same cardinality
                cluster_idx     = cluster.scoring_units
                non_cluster_idx = list(set(layer_idx).difference(cluster_idx))
                external_idx    = list(np.random.choice(non_cluster_idx, len(cluster_idx), replace=False))
                
                activations_idx = {
                    'cluster' : cluster_idx,
                    'external': external_idx
                }

            # 2) Random - use scattered random units with the same dimensionality of the cluster
            case 'random': 
                
                scoring_units = list(np.random.choice(
                    np.arange(0, tot_obj), 
                    size=clu_obj, 
                    replace=False
                ))
            
            # 3) Random adjacent - use continuos random units with the same dimensionality of the cluster
            case 'random_adj':

                start= np.random.randint(0, tot_obj - clu_obj + 1)
                scoring_units = list(range(start, start + clu_obj))
                
            # 4) Scoring only subset of the cluster
            case 'subset' | 'subset_top' | 'subset_bot' | 'subset_rand':
                
                # Extract the optimizer units
                opt_units: List[int] = [int(idx) for idx in clu_conf[str(Args.OptimUnits)].split()]
                
                # For subset specific indexes they are the specific indexes
                # while for the other ones it just the number of units and it is a single number
                # TODO Raise specific errors
                
                # Check optimizing units consistency
                for opt_unit in opt_units:
                    assert 1 <= opt_unit <= len(cluster)
                
                # In the case of subset opt_units is a list of indexes
                # otherwise it is a single number which we extract
                if scr_type != 'subset':
                    assert len(opt_units) == 1
                    opt_units_ = opt_units[0]
                    
                # Get cluster indexes and sort them by score
                # From high centrality to lower one
                clu_idx: ScoringUnit = cluster.scoring_units  # NOTE: They are already sorted by descending rank
                
                clu_opt_idx: ScoringUnit
                match clu_conf[str(Args.ScoringType)]:
                    
                    case 'subset'     : clu_opt_idx = [clu_idx[opt_unit-1] for opt_unit in opt_units]             # Use specified list of index (off-by-one)
                    case 'subset_top' : clu_opt_idx = clu_idx[:opt_units_]                                        # Use top-k units
                    case 'subset_bot' : clu_opt_idx = clu_idx[(-1*opt_units_):]                                   # Use bot-k units
                    case 'subset_rand': clu_opt_idx = list(np.random.choice(clu_idx, opt_units_, replace=False))  # Sample k units at random
                
                # Retrieve the non scoring units
                clu_non_opt_idx = list(set(clu_idx).difference(clu_opt_idx))
                
                # Compute the non-cluster units as difference
                non_cluster_idx = list(set(layer_idx).difference(clu_idx))
                
                # Extract as many non cluster optimized units outside the cluster at random
                ext_non_opt_idx = list(np.random.choice(non_cluster_idx, len(clu_non_opt_idx), replace=False))
                
                scoring_units = clu_opt_idx
                
                activations_idx = {
                    "Cluster optimized"      : clu_opt_idx,
                    "External non optimized" : ext_non_opt_idx,
                    "Cluster non optimized"  : clu_non_opt_idx
                }

            # Default - raise an error
            case _:
                err_msg =  f'Invalid `scr_type`: {clu_conf[str(Args.ScoringType)]}. '\
                            'Choose one between {cluster, random, random_adj, subset_top, subset_bot, fraction rand}. '
                raise ValueError(err_msg)  

        # --- SCORER ---

        scorer = ActivityScorer(
            scoring_units={layer_name: scoring_units},
            units_reduction=units_reduction,
            layer_reduction=scr_conf[str(Args.LayerReduction)],
            units_map=units_mapping
        )

        # --- OPTIMIZER ---
        
        optim = CMAESOptimizer(
            codes_shape  = generator.input_dim,
            rnd_seed     =     conf[str(Args.RandomSeed)],
            rnd_distr    = opt_conf[str(Args.RandomDistr)],
            rnd_scale    = opt_conf[str(Args.RandomScale)],
            pop_size     = opt_conf[str(Args.PopulationSize)],
            sigma0       = opt_conf[str(Args.Sigma0)]
        )
        
        # optim = GeneticOptimizer(
        #     codes_shape  = generator.input_dim,
        #     rnd_seed     =     conf[str(Args.RandomSeed)],
        #     rnd_distr    = opt_conf[str(Args.RandomDistr)],
        #     rnd_scale    = opt_conf[str(Args.RandomScale)],
        #     pop_size     = opt_conf[str(Args.PopulationSize)]   
        # )

        #  --- LOGGER --- 

        log_conf[str(Args.ExperimentTitle)] = ClusteringOptimizationExperiment.EXPERIMENT_TITLE
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
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=(800, 800))
                )

                # Add screen for natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(800, 800))
                    )

        # --- DATA ---
        
        data = {
            'render'         : conf[str(Args.Render)],
            'use_nat'        : use_nat,
            'close_screen'   : conf.get('close_screen', False),
            'activation_idx' : activations_idx
        }

        # Experiment configuration
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            iteration      = conf[str(Args.NumIterations)],
            nat_img_loader = nat_img_loader,
            data           = data, 
            name           = log_conf[str(Args.ExperimentName)]
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
        self._render         = cast(bool, data['render'])
        self._close_screen   = cast(bool, data['close_screen'])
        self._use_nat        = cast(bool, data['use_nat'])
        
        self._activation_idx = cast(Dict[str, ScoringUnit], data['activation_idx'])

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:

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

    def _init(self) -> ZdreamMessage:

        msg = super()._init()
        
        # Data structure to save clusters activations
        self._activations: Dict[str, List[NDArray]] = {k: [] for k in self._activation_idx}

        # Data structure to save best score and best image
        if self._use_nat:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
        
        return msg

    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.solution
        best_synthetic = self.generator(codes=best_code)
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

    def _finish(self, msg : ZdreamMessage):

        super()._finish(msg)

        # Close screens
        if self._close_screen:
            self._logger.close_all_screens()
            
        # 1) SAVING BEST STIMULI

        # Save visual stimuli (synthetic and natural)
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.solution)

        # We retrieve the stored best natural image
        if self._use_nat:
            best_nat = self._best_nat_img

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
            
        # 2) PLOT
        if len(self._activations):
            
            # Save visual stimuli (synthetic and natural)
            plot_dir = make_dir(path=path.join(self.dir, 'plots'), logger=self._logger)
            
            # Stack arrays
            activations: Dict[str, NDArray] = {name: np.stack(acts) for name, acts in self._activations.items()}
            
            # Plot
            plot_activations(
                activations=activations,
                out_dir=plot_dir,
                logger=self._logger
            )
        
        self._logger.info(mess='')
        
        return msg
    
    # --- PIPELINE ---
    
    def _states_to_scores(self, data: Tuple[States, ZdreamMessage]) -> Tuple[Scores, ZdreamMessage]:
        
        states, msg = data
        
        assert len(states) == 1
        full_activations = list(states.values())[0][msg.mask]
        
        # Compute mean activations for each group
        for act_name, act_idx in self._activation_idx.items():
            activation = np.mean(full_activations[:, act_idx], axis=0)
            self._activations[act_name].append(activation)
        
        return super()._states_to_scores(data)

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

class _ClusteringOptimizationMultiExperiment(MultiExperiment):

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
                display_size=(1200, 1200)
            )
        ]

        # Add screen for natural images if at least one will use it
        use_nat = any(
            parse_boolean_string(conf['natural_stimuli_loader'][str(Args.Template)]).count(False) > 0 
            for conf in self._search_config
        )
        
        if use_nat:
            screens.append( DisplayScreen(title=ClusteringOptimizationExperiment.NAT_IMG_SCREEN, display_size=(400, 400)) )

        return screens
    
    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger
    

class UnitsWeightingMultiExperiment(_ClusteringOptimizationMultiExperiment):
        
    def _init(self):
        
        super()._init()
        
        self._data['desc'       ] = 'Comparison between cluster-weighted an arithmetic score average'
        self._data['cluster_idx'] = list()  # Cluster idx in the clustering
        self._data['scores'     ] = list()  # Scores across generation
        self._data['weighted'   ] = list()  # Boolean flag if scoring was weighted or not

    def _progress(
        self, 
        exp: ClusteringOptimizationExperiment, 
        conf: Dict[str, Any],
        msg : ZdreamMessage, 
        i: int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        clu_conf = conf['clustering']
        
        self._data['scores']      .append(msg.scores_gen_history)
        self._data['cluster_idx'].append(clu_conf[str(Args.ClusterIdx)])
        self._data['weighted']   .append(clu_conf[str(Args.WeightedScore)])    

    def _finish(self):
        
        super()._finish()
        
        plot_weighted(
            cluster_idx  = self._data['cluster_idx'],
            weighted     = self._data['weighted'],
            scores       = self._data['scores'],
            out_dir      = self.target_dir,
            logger       = self._logger
        )


class ClusteringScoringTypeMultiExperiment(_ClusteringOptimizationMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'       ] = 'Comparison between clustering units optimization'
        
        # Cluster idx: scr_type: scores
        self._data['scr_type'] = defaultdict(lambda: defaultdict(list))  


    def _progress(
        self, 
        exp: ClusteringOptimizationExperiment,
        conf: Dict[str, Any],
        msg : ZdreamMessage, 
        i: int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        clu_conf = conf['clustering']
        
        clu_idx  = clu_conf[str(Args.ClusterIdx)]
        scr_type = clu_conf[str(Args.ScoringType)]
        score    = float(msg.stats_gen['best_score'])
        
        self._data['scr_type'][clu_idx][scr_type].append(score)

    def _finish(self):
        
        # Dictionary redefinition with no lambdas
        # SEE: https://stackoverflow.com/questions/72339545/attributeerror-cant-pickle-local-object-locals-lambda
        
        self._data['scr_type'] = {
            cluster_idx: {
                scr_type: [score for score in scores]
                for scr_type, scores in scr.items()
            }
            for cluster_idx, scr in self._data['scr_type'].items()
        }
        
        super()._finish()
        
        plot_scr(
            data    = self._data['scr_type'],
            out_dir = self.target_dir,
            logger  = self._logger
        )


class ClusteringSubsetOptimizationMultiExperiment(_ClusteringOptimizationMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'] = 'Comparison between co-activation inside and outside cluster for non optimized units'
        
        # Dictionary: cluster-idx: optimized_unit: component: list of sample final average activation
        self._data['cluster_activations'] = defaultdict(lambda: defaultdict(list))


    def _progress(
        self, 
        exp: ClusteringOptimizationExperiment, 
        conf: Dict[str, Any],
        msg : ZdreamMessage,
        i: int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        clu_conf = conf['clustering']
        
        assert clu_conf[str(Args.ScoringType)].startswith('subset')
        
        cluster_idx  = clu_conf[str(Args.ClusterIdx)]
        cluster_unit = int(clu_conf[str(Args.OptimUnits)])
        
        # Compute average of final activation
        # as the average of the last column
        avg_activation = {
            name: float(np.mean(activations[-1]))
            for name, activations in exp._activations.items()
        }
        
        self._data['cluster_activations'][cluster_idx][cluster_unit].append(avg_activation)

    def _finish(self):
        
        # Dictionary redefinition with no lambdas
        # SEE: https://stackoverflow.com/questions/72339545/attributeerror-cant-pickle-local-object-locals-lambda
        
        self._data['cluster_activations'] = {
            cluster_idx: {
                cluster_unit: values
                for cluster_unit, values in units.items()
            }
            for cluster_idx, units in self._data['cluster_activations'].items()
        }
        
        super()._finish()
        
        plot_dir = path.join(self.target_dir, 'plots')
        self._logger.info(mess=f'Saving plots to {plot_dir}')
        self._logger.formatting = lambda x: f'> {x}'
        plot_subsetting_optimization(
            clusters_activations=self._data['cluster_activations'],
            logger=self._logger,
            out_dir=self.target_dir
        )
        
        self._logger.info(mess='')


class ClustersBestStimuliMultiExperiment(_ClusteringOptimizationMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'] = 'Grid of best images for each element in the cluster. '
        
        # Dictionary: cluster-idx: optimized_unit: (score, code)
        self._data['best_codes'] = defaultdict(lambda: defaultdict(Tuple[float, Codes]))


    def _progress(
        self, 
        exp: ClusteringOptimizationExperiment, 
        conf: Dict[str, Any],
        msg : ZdreamMessage,
        i: int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        clu_conf = conf['clustering']
        
        cluster_idx  = clu_conf[str(Args.ClusterIdx)]
        cluster_unit = int(clu_conf[str(Args.OptimUnits)])
        
        best_score : float = msg.stats_gen['best_score']
        best_code  : Codes = msg.solution[0]
        
        cluster_bests = self._data['best_codes'][cluster_idx]
        
        if cluster_unit not in cluster_bests:
        
            cluster_bests[cluster_unit] = best_score, best_code
        
        else:
            
            clu_best_score, _ = cluster_bests[cluster_unit]
            
            if best_score > clu_best_score:
                cluster_bests[cluster_unit] = best_score, best_code


    def _finish(self):
        
        # Dictionary redefinition with no lambdas
        # SEE: https://stackoverflow.com/questions/72339545/attributeerror-cant-pickle-local-object-locals-lambda
        
        self._data['best_codes'] = {
            cluster_idx: {
                cluster_unit: best
                for cluster_unit, best in units.items()
            }
            for cluster_idx, units in self._data['best_codes'].items()
        }
        
        super()._finish()
        
        stimuli_dir = path.join(self.target_dir, 'stimuli')
        self._logger.info(mess=f'Saving stimuli to {stimuli_dir}')
        os.makedirs(stimuli_dir)
        
        gen_conf = self._search_config[0]['generator']
        
        generator = DeePSiMGenerator(
            root    = gen_conf[str(Args.GenWeights)],
            variant = gen_conf[str(Args.GenVariant)]
        )
        
        self._logger.formatting = lambda x: f'> {x}'
        plot_cluster_best_stimuli(
            cluster_codes=self._data['best_codes'],
            generator=generator,
            logger=self._logger,
            out_dir=stimuli_dir
        )
        self._logger.reset_formatting()
        
        self._logger.info(mess='')

class ClusterStimuliTargetMultiExperiment(_ClusteringOptimizationMultiExperiment):
    
    def _init(self):
        
        super()._init()
        
        self._data['desc'] = 'Best stimuli for all images of one neuron both using arithmetic or weighted average. '
        
        # Dictionary: cluster-idx: weighting_type: (score, code)
        self._data['best_codes'] = defaultdict(lambda: defaultdict(Tuple[float, Codes]))


    def _progress(
        self, 
        exp: ClusteringOptimizationExperiment, 
        conf: Dict[str, Any],
        msg : ZdreamMessage,
        i: int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        clu_conf = conf['clustering']
        
        cluster_idx        = clu_conf[str(Args.ClusterIdx)]
        cluster_weighting  = clu_conf[str(Args.WeightedScore)]
        
        best_score : float = msg.stats_gen['best_score']
        best_code  : Codes = msg.solution[0]
        
        cluster_bests = self._data['best_codes'][cluster_idx]
        
        if cluster_weighting not in cluster_bests:
        
            cluster_bests[cluster_weighting] = best_score, best_code
        
        else:
            
            clu_best_score, _ = cluster_bests[cluster_weighting]
            
            if best_score > clu_best_score:
                cluster_bests[cluster_weighting] = best_score, best_code


    def _finish(self):
        
        # Dictionary redefinition with no lambdas
        # SEE: https://stackoverflow.com/questions/72339545/attributeerror-cant-pickle-local-object-locals-lambda
        
        self._data['best_codes'] = {
            weighted_score: {
                cluster_unit: best
                for cluster_unit, best in units.items()
            }
            for weighted_score, units in self._data['best_codes'].items()
        }
        
        super()._finish()
        
        stimuli_dir = path.join(self.target_dir, 'stimuli')
        self._logger.info(mess=f'Saving stimuli to {stimuli_dir}')
        os.makedirs(stimuli_dir)
        
        gen_conf = self._search_config[0]['generator']
        
        generator = DeePSiMGenerator(
            root    = gen_conf[str(Args.GenWeights)],
            variant = gen_conf[str(Args.GenVariant)]
        ).to(device)
        
        self._logger.formatting = lambda x: f'> {x}'
        
        plot_cluster_target(
            cluster_codes=self._data['best_codes'],
            generator=generator,
            logger=self._logger,
            out_dir=stimuli_dir
        )
        self._logger.reset_formatting()
        
        self._logger.info(mess='')