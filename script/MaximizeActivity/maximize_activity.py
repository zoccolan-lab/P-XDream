import os
from os import path
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image
from numpy.typing import NDArray
import torch
from torchvision.transforms.functional import to_pil_image

from script.MaximizeActivity.plotting import multiexp_lineplot, plot_optimizing_units, plot_scores, plot_scores_by_label
from script.utils.cmdline_args import Args
from script.utils.parsing import parse_boolean_string, parse_recording, parse_scoring
from script.utils.misc import make_dir
from zdream.experiment import ZdreamExperiment, MultiExperiment
from zdream.generator import Generator, DeePSiMGenerator
from zdream.optimizer import CMAESOptimizer, Optimizer
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, TorchNetworkSubject
from zdream.utils.dataset import ExperimentDataset, MiniImageNet, NaturalStimuliLoader
from zdream.utils.io_ import to_gif
from zdream.utils.logger import DisplayScreen, Logger, LoguruLogger
from zdream.utils.message import ZdreamMessage
from zdream.utils.misc import concatenate_images, device
from zdream.utils.probe import InfoProbe, RecordingProbe
from zdream.utils.types import Codes, RecordingUnit, ScoringUnit, Stimuli, Scores, States


class MaximizeActivityExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "MaximizeActivity"

    # Screen titles for the display
    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # --- OVERRIDES ---
    
    # We override the properties to cast them to the specific components to activate their methods

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 
    
    # --- CONFIG ---

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'MaximizeActivityExperiment':
        '''
        Create a MaximizeActivityExperiment instance from a configuration dictionary.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: MaximizeActivity
        '''

        # Extract specific configurations
        gen_conf = conf['generator']
        nsl_conf = conf['natural_stimuli_loader']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']
        
        # Set numpy random seed
        np.random.seed(conf['random_seed'])

        # --- NATURAL IMAGE LOADER ---

        # Parse template and derive if to use natural images
        template = parse_boolean_string(boolean_str=nsl_conf[str(Args.Template)])
        use_nat  = template.count(False) > 0
        
        # Create dataset and loader
        dataset  = MiniImageNet(root=nsl_conf[str(Args.Dataset)])
        nat_img_loader   = NaturalStimuliLoader(
            dataset=dataset,
            template=template,
            shuffle=nsl_conf[str(Args.Shuffle)]
        )
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root=gen_conf[str(Args.GenWeights)],
            variant=gen_conf[str(Args.GenVariant)]
        )
        
        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=sbj_conf[str(Args.NetworkName)]).layer_info

        # Probe
        record_target = parse_recording(input_str=sbj_conf[str(Args.RecordingLayers)], net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=sbj_conf[str(Args.NetworkName)],
        )
        
        # Set the network in evaluation mode
        sbj_net.eval()

        # --- SCORER ---

        # Parse target neurons
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

        optim = CMAESOptimizer(
            codes_shape=generator.input_dim,
            rnd_seed  =     conf[str(Args.RandomSeed)],
            pop_size  = opt_conf[str(Args.PopulationSize)],
            rnd_distr = opt_conf[str(Args.RandomDistr)],
            rnd_scale = opt_conf[str(Args.RandomScale)],
            sigma0    = opt_conf[str(Args.Sigma0)],
        )

        #  --- LOGGER --- 

        log_conf[str(Args.ExperimentTitle)] = MaximizeActivityExperiment.EXPERIMENT_TITLE
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

                # Add screen fro natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(400, 400))
                    )

        # --- DATA ---
        
        data = {
            "dataset": dataset if use_nat else None,
            'render': conf[str(Args.Render)],
            'close_screen': conf.get('close_screen', False),
            'use_nat': use_nat
        }

        # --- CREATE INSTANCE ---
        
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            nat_img_loader = nat_img_loader,
            iteration      = conf[str(Args.NumIterations)],
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
        nat_img_loader: NaturalStimuliLoader,
        data:           Dict[str, Any] = dict(),
        name:           str = 'maximize_activity'
    ) -> None:
        ''' 
        Uses the same signature as the parent class ZdreamExperiment.
        It save additional information from the `data`attribute to be used during the experiment.
        '''

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

        # Save attributes from data
        self._render        = cast(bool, data[str(Args.Render)])
        self._close_screen  = cast(bool, data['close_screen'])
        self._use_natural   = cast(bool, data['use_nat'])
        
        # Save dataset if used
        if self._use_natural:
            self._dataset = cast(ExperimentDataset, data['dataset'])
            
    # --- DATA FLOW ---
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        Override the method to save the last set of stimuli and the labels if natural images are used.
        '''

        # We save the last set of stimuli in `self._stimuli`
        self._stimuli, msg = data
        
        # We save the last set of labels if natural images are used
        if self._use_natural:
            self._labels.extend(msg.labels)
        
        states, msg = super()._stimuli_to_states(data)

        return states, msg

    def _scores_to_codes(self, data: Tuple[Scores, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        ''' We use scores to save stimuli (both natural and synthetic) that achieved the highest score. '''

        sub_score, msg = data

        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._scores_to_codes((sub_score, msg))
    
    # --- RUN METHODS ---

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:
        ''' Add information about the best score and the average score.'''
        
        # Synthetic scores
        stat_gen = msg.stats_gen
        
        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        
        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'
        
        # Best natural score
        if self._use_natural:
            
            stat_nat = msg.stats_nat
            best_nat = cast(NDArray, stat_nat['best_score']).mean()
            best_nat_str = f'{best_nat:.1f}'
            
            desc = f'{desc} | {best_nat_str}'

        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super}{desc}'

    def _init(self) -> ZdreamMessage:
        ''' We initialize the experiment and add the data structure to save the best score and best image. '''

        msg = super()._init()

        # Data structures to save best natural images and labels presented
        if self._use_natural:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
            self._labels: List[int] = []
        
        # Gif
        self._gif: List[Image.Image] = []

        return msg

    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.solution
        best_synthetic = self.generator(codes=best_code)
        
        best_synthetic_img = to_pil_image(best_synthetic[0])
        
        if self._use_natural:
            best_natural = self._best_nat_img
            
        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )          
            
        if self._render:

            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )

            if self._use_natural:
                self._logger.update_screen(
                    screen_name=self.NAT_IMG_SCREEN,
                    image=to_pil_image(best_natural)
                )   

    def _finish(self, msg : ZdreamMessage):
        ''' 
        Save best stimuli and make plots
        '''

        super()._finish(msg)

        if self._close_screen:
            self._logger.close_all_screens()
            
        # 1. SAVE VISUAL STIMULI (SYNTHETIC AND NATURAL)

        # Create image folder
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.solution)
        best_gen = best_gen[0] # remove 1 batch size
        
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen), 'best synthetic')]
        
        # If used we retrieve the best natural image
        if self._use_natural:
            best_nat = self._best_nat_img
            to_save.append((to_pil_image(best_nat), 'best natural'))
            to_save.append((concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli'))
        
        # Saving images
        for img, label in to_save:
            
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)
        
        # Saving gif
        out_fp = path.join(img_dir, 'evolving_best.gif')
        self._logger.info(f'> Saving evolving best stimuli across generations to {out_fp}')
        to_gif(image_list=self._gif, out_fp=out_fp)

        self._logger.info(mess='')
        
        # 2. SAVE PLOTS

        # Create plots folder
        plots_dir = make_dir(path=path.join(self.dir, 'plots'), logger=self._logger)
        
        self._logger.formatting = lambda x: f'> {x}'
        
        # Plot scores
        plot_scores(
            scores=(
                np.stack(msg.scores_gen_history),
                np.stack(msg.scores_nat_history) if self._use_natural else np.array([])
            ),
            stats=(
                msg.stats_gen,
                msg.stats_nat if self._use_natural else dict(),
            ),
            out_dir=plots_dir,
            logger=self._logger
        )

        # Plot scores by category
        if self._use_natural:
            plot_scores_by_label(
                scores=(
                    np.stack(msg.scores_gen_history),
                    np.stack(msg.scores_nat_history)
                ),
                lbls    = self._labels,
                out_dir = plots_dir, 
                dataset = self._dataset,
                logger=self._logger
            )
        
        self._logger.reset_formatting()
        self._logger.info(mess='')
        
        
        return msg
    
class MaximizeActivityExperiment2(MaximizeActivityExperiment):
    '''
    This class extends the MaximizeActivityExperiment to save activation states
    '''
    
    def __init__(
        self,
        generator: Generator, 
        subject: InSilicoSubject, 
        scorer: Scorer, 
        optimizer: Optimizer, 
        iteration: int, 
        logger: Logger, 
        nat_img_loader: NaturalStimuliLoader, 
        data: Dict[str, Any] = dict(), 
        name: str = 'maximize_activity'
    ) -> None:
        
        super().__init__(generator, subject, scorer, optimizer, iteration, logger, nat_img_loader, data, name)
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        Override the method to save the activation states of the network.
        '''
        
        states, msg = super()._stimuli_to_states(data=data)

        # We save the states history
        # NOTE: This can be memory consuming 
        msg.states_history.append(states)
        
        return states, msg
    
class MaximizeActivityRFMapsExperiment(MaximizeActivityExperiment):
    '''
    This class extends the MaximizeActivityExperiment to save the receptive fields of the network.
    '''
    
    def _init(self) -> ZdreamMessage:
        
        msg = super()._init()
        
        CONV_THRESHOLD = {
            'alexnet': 15
        }
        
        INPUT = '00_input_01'
        
        # Create mock images for receptive field mapping (both forward and backward)
        #self.nr_imgs4rf = 10
        mock_inp = torch.randn(1, *self.subject.in_shape[1:], device=device, requires_grad=True)
        
        # Get recording layers recording from. 
        # We will map the scoring neurons on each of them.
        # We sort the target layers by their depth in ascending order
        
        rec_layers = [INPUT] + msg.rec_layers
        rec_layers = sorted(rec_layers, key=lambda x: int(x.split('_')[0]))
        
        # NOTE: Following operations works only if scoring from a single layer 
        
        rec_targets: Dict[str, RecordingUnit] = self.subject.target
        
        # NOTE: In case of conv layers, scored_units values will be a np.array of shape 3 x n
        # where each row corresponds to the coordinates of the unit of interest (c x h x w)
        
        conv_threshold = CONV_THRESHOLD[self.subject.name]
        
        scored_units = {
            layer : 
                np.expand_dims(np.array(units), axis=0) 
                if int(layer.split('_')[0]) > conv_threshold  # check if it not a conv layer 
                else np.row_stack([rec_targets[layer][i][units] for i in range(len(rec_targets[layer]))]) # type: ignore
            for layer, units in msg.scr_units.items()
        }

        
        # Create the InfoProbe and attach it to the subject
        mapped_layers   = list(set(rec_layers) - set(scored_units.keys()))
        backward_target = {ml: scored_units for ml in mapped_layers}
        
        probe = InfoProbe(
            inp_shape=self.subject._inp_shape,
            rf_method='backward',
            backward_target=backward_target  # type: ignore
        )

        # NOTE: For backward receptive field we need to register both
        #       the forward and backward probe hooks
        self.subject.register(probe)

        # Expose the subject to the mock input to collect the set
        # of shapes of the underlying network
        _ = self.subject(stimuli=mock_inp, raise_no_probe=False, with_grad=True)
        
        # Collect the receptive fields from the info probe
        msg.rf_maps = probe.rec_field
        
        # Get rf perimeter on the input
        # NOTE: they are computed on network input size, that can differ
        #       from the outputted generated images (therefore, a resizing of rfs is implemented)
        rf_on_input = msg.rf_maps[(INPUT, msg.scr_layers[-1])]

        # Create a mask for the 2D perimeter of the receptive fields
        rf_2p_mask = [
            np.zeros(self.generator.output_dim[-2:], dtype=bool) 
            for _ in range(len(rf_on_input))
        ]
        
        # Rescale the rf perimeter on the input to the generated image
        img_side_gen       = self.generator.output_dim[-1]
        img_side_net_input = self.subject.in_shape[-1]
        rescale_factor     = img_side_gen / img_side_net_input
        
        # TODO @Lorenzo
        for i, rf in enumerate(rf_on_input):
            
            # TODO @Lorenzo
            rf_rescaled = [
                tuple(
                    round(c * rescale_factor) 
                    if round(c * rescale_factor) < img_side_gen 
                    else (img_side_gen-1) 
                    for c in coord
                ) 
                if idx == 1 or idx == 2
                else coord
                for idx, coord in enumerate(rf)
            ]
            
            # TODO @Loenzo       
            rf_2p_mask[i][rf_rescaled[1][ 0]:rf_rescaled[1][-1]+1, rf_rescaled[2][0]]                      = True
            rf_2p_mask[i][rf_rescaled[1][ 0]:rf_rescaled[1][-1]+1, rf_rescaled[2][-1]]                     = True
            rf_2p_mask[i][rf_rescaled[1][ 0],                      rf_rescaled[2][0]:rf_rescaled[2][-1]+1] = True
            rf_2p_mask[i][rf_rescaled[1][-1],                      rf_rescaled[2][0]:rf_rescaled[2][-1]+1] = True
        
        # Save mask
        self._rf_2p_mask = rf_2p_mask
        
        # Remove the probe from the subject
        self.subject.remove(probe)
        
        return msg
        
    def _finish(self, msg: ZdreamMessage):
        
        msg = super()._finish(msg)
        
        # SAVE VISUAL STIMULI (SYNTHETIC AND NATURAL) with their receptive fields

        # Create image folder
        img_dir = path.join(self.dir, 'images') # already existing

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.solution)
        best_gen = best_gen[0] # remove 1 batch size
        
        # Apply mask
        for rf_mask in self._rf_2p_mask:
            best_gen[:, :, rf_mask] = np.inf
        
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen), 'best synthetic RF')]
        
        # If used we retrieve the best natural image
        if self._use_natural:
            best_nat = self._best_nat_img
            
            for rf_mask in self._rf_2p_mask:
                best_nat[:, :, rf_mask] = np.inf
            
            to_save.append((to_pil_image(best_nat), 'best natural RF'))
            to_save.append((concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli RF'))
        
        # Saving images
        for img, label in to_save:
            
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)


# --- MULTI EXPERIMENT ---

class _MaximizeActivityMultiExperiment(MultiExperiment):
    ''' Generic class handling different multi-experiment types. '''
    
    def __init__(
            self, 
            experiment:      Type['MaximizeActivityExperiment'], 
            experiment_conf: Dict[str, List[Any]], 
            default_conf:    Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)

        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True
        
    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger

    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [ DisplayScreen(title=MaximizeActivityExperiment.GEN_IMG_SCREEN, display_size=(400, 400)) ]

        # Add screen for natural images if at least one will use it
        use_nat = any(
            parse_boolean_string(conf['natural_stimuli_loader']['template']).count(False) > 0 
            for conf in self._search_config
        )
        
        if use_nat:
            screens.append( DisplayScreen(title=MaximizeActivityExperiment.NAT_IMG_SCREEN, display_size=(400, 400)) )

        return screens

class NeuronScalingMultiExperiment(_MaximizeActivityMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'   ] = 'Scores at varying number of scoring neurons' # TODO
        self._data['score'  ] = list()
        self._data['neurons'] = list()
        self._data['layer'  ] = list()
        self._data['iter'   ] = list()


    def _progress(
        self, 
        exp  : MaximizeActivityExperiment, 
        conf : Dict[str, Any],
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        tot_units = sum([rec_units[0].shape[0] for rec_units in msg.rec_units.values()]) # type: ignore

        self._data['score']  .append(msg.stats_gen['best_score'])
        self._data['neurons'].append(tot_units)
        self._data['layer']  .append(msg.scr_layers)
        self._data['iter']   .append(exp.iteration)

    def _finish(self):
        
        super()._finish()

        plot_optimizing_units(
            multiexp_data=self._data,
            out_dir=self._logger.dir,
            logger=self._logger
        )


class LayersCorrelationMultiExperiment(_MaximizeActivityMultiExperiment):

    def _init(self):

        super()._init()
        
        # Description
        self._data['desc'] = 'Layers correlations ...' # TODO @Lorenzo

        # Delta activation
        self._data['deltaA_rec'] = list()

        # Hyperparameter and recordings
        self._data['layer']     = list()
        self._data['score']     = list()
        self._data['score_nat'] = list()
        self._data['iter']      = list()
        self._data['neurons']   = list()

        # Correlations
        self._data['corr_opt_rec_mean'] = list()
        self._data['corr_opt_rec_var']  = list()
        self._data['corr_rec_rec_mean'] = list()
        self._data['corr_rec_rec_var']  = list()

    
    def _progress(
        self, 
        exp  : MaximizeActivityExperiment2, 
        conf : Dict[str, Any],
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)

        # Update score and parameters
        self._data['score']    .append(msg.stats_gen['best_score'])
        self._data['score_nat'].append(msg.stats_nat['best_score'] if exp._use_natural else np.nan)
        self._data['neurons']  .append(0) # TODO
        self._data['layer']    .append(msg.scr_layers)
        self._data['iter']     .append(exp.iteration)
        
        # Create new dictionary for fields
        deltaA_rec        = {}
        corr_opt_rec_mean = {}
        corr_opt_rec_var  = {}
        corr_rec_rec_mean = {}
        corr_rec_rec_var  = {}

        # Iterate over recorded layers to get statistics about non optimized sites
        non_scoring = self._get_non_scoring_units(exp=exp)
        
        print("AAAA")
        print(msg.states_history)
        
        states_history = {
            key: np.stack([state[key] for state in msg.states_history]) if msg.states_history else np.array([])
            for key in msg.rec_layers
        }
        
        for layer, activations in states_history.items():

            # Extract non score units for that layer
            non_scoring_units = non_scoring[layer]

            # If there are units recorder but not scored
            if non_scoring_units:

                # Get the average difference in activation between the last and the first optim iteration
                deltaA_rec[layer] = np.mean(activations[-1, :, non_scoring_units]) - \
                                    np.mean(activations[ 0, :, non_scoring_units])   

                # Compute the correlation between each avg recorded unit and the mean of the optimized sites. 
                # Store correlation mean and variance
                avg_rec  = np.mean(activations[:, :, non_scoring_units], axis=1).T
                corr_vec = np.corrcoef(msg.stats_gen['mean_gens'], avg_rec)[0, 1:]
                
                corr_opt_rec_mean[layer] = np.mean(corr_vec)
                corr_opt_rec_var [layer] = np.std(corr_vec)

                # Compute the correlation between each avg recorded units. 
                # Store correlation mean and variance
                corr_rec_rec_mat = np.corrcoef(avg_rec)
                upper_triangle   = np.triu_indices(corr_rec_rec_mat.shape[0], k=1)
                
                corr_rec_rec_mean[layer] = np.mean(corr_rec_rec_mat[upper_triangle])
                corr_rec_rec_var[layer]  = np.std (corr_rec_rec_mat[upper_triangle])

        # Update correlations
        self._data['deltaA_rec']       .append(deltaA_rec)
        self._data['corr_opt_rec_mean'].append(corr_opt_rec_mean)
        self._data['corr_opt_rec_var'] .append(corr_opt_rec_var)
        self._data['corr_rec_rec_mean'].append(corr_rec_rec_mean)
        self._data['corr_rec_rec_var'] .append(corr_rec_rec_var)

    def _finish(self):

        super()._finish()

        # Organize results as a dataframe
        df = self._create_df(multiexp_data=self._data)

        # Create output image folder
        plots_dir = make_dir(path=path.join(self.target_dir, 'plots'), logger=self._logger)

        self._logger.formatting = lambda x: f'> {x}'

        # Plot neuron score scaling
        sc_metrics = ['scores', 'scores_norm']
        
        for m in sc_metrics:
            
            multiexp_lineplot(
                out_df=df, 
                gr_vars=['layers', 'neurons'],
                out_dir=plots_dir,
                y_var=m,
                metrics=['mean', 'sem'],
                logger=self._logger
            )
        
        # Plot different information varying the response variable
        for c in df.columns:

            if 'rec' in c:

                multiexp_lineplot(
                    out_df=df, 
                    gr_vars= ['layers', 'neurons'],
                    out_dir=plots_dir,
                    y_var = c, 
                    metrics = ['mean', 'sem'],
                    logger=self._logger
                )
        
        self._logger.reset_formatting
        
    @staticmethod
    def _create_df(multiexp_data: Dict[str, Any]) -> DataFrame:
        '''
        Reorganize multi-experiment result in a Dataframe for plotting.

        NOTE: This static method allows for an offline computation provided
            multi-experiment results stored as .PICKLE
        '''
        
        def extract_layer(el):
            ''' Auxiliar function to extract layer names from a list element'''
            if len(el) > 1:
                raise ValueError(f'Expected to record from a single layer, multiple where found: {el}')
            return el[0]
        
        # Organize main experiment data
        data = {
            'layers'  : np.stack([extract_layer(el) for el in multiexp_data['layer']]),
            'neurons' : np.stack(multiexp_data['neurons'], dtype=np.int32),
            'iter'    : np.stack(multiexp_data['iter'], dtype=np.int32),
            'scores'  : np.concatenate(multiexp_data['score']),
            'scores_nat': np.concatenate(multiexp_data['score_nat'])     
        }

        data['scores_norm'] = data['scores']/data['scores_nat']
        
        # Extract recording-related data 

        # Unique_keys referred to a 
        unique_layers = list(set().union(*[d.keys() for d in multiexp_data['deltaA_rec']]))

        # Iterate over the keys involved in recordings
        rec_keys = ['deltaA_rec', 'corr_opt_rec_mean', 'corr_opt_rec_var', 'corr_rec_rec_mean', 'corr_rec_rec_var']

        for key in rec_keys:

            # Key prefix depends on th
            prefix = 'Δrec_' if key == 'deltaA_rec' else key

            # Initialize dictionary with Nans
            rec_dict = {
                f'{prefix}{layer}': np.full(len(multiexp_data[key]), np.nan, dtype=np.float32) 
                for layer in unique_layers
            }
            
            # Add result to specific experiment if layer name match
            for i, result in enumerate(multiexp_data[key]):
                for layer in unique_layers:
                    if layer in result.keys():
                        rec_dict[f'{prefix}{layer}'][i] = result[layer]
                        
            # Unify all data into a single dictionary
            data.update(rec_dict)
        
        # Transform data in dictionary form
        return pd.DataFrame(data)

    @staticmethod
    def _get_non_scoring_units(exp: MaximizeActivityExperiment) -> Dict[str, ScoringUnit]:
        '''
        Given an experiment returns a mapping between layer name and activations indexes
        for recorded but non-scored units
        '''

        non_scoring_units = {}

        for layer, rec_units in exp.subject.target.items():
        
            # If not all recorded extract their length
            if rec_units:
                recorded = rec_units[0].size  # length of the first array of the tuple
            
            # If all recorded is the total number of neurons
            else:
                recorded = np.prod(exp.subject.layer_info[layer])
            
            # Compute the non recorded units
            try:
                scoring = exp.scorer.scoring_units[layer]
            except KeyError:
                scoring = []
            
            not_scoring = set(range(recorded)).difference(scoring)

            non_scoring_units[layer] = list(not_scoring)

        return non_scoring_units

"""
@Lorenzo

#-- RANDOM UNITS SAMPLING ---

        #QUA FARE IL SAMPLING DELLE UNITà RANDOMICHE CHE POI POSSONO ESSERE USATE (SCORING) PER IL RECORDING
        
        # Create a on-the-fly network subject to extract all network layer names
        #ASSUMPION: ALL THE EXPERIMENTS SHARE THE SAME NETWORK MODEL (e.g. alexnet)
        net_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(network_name=self._search_config[0]['subject']['net_name']).layer_info
        layer_names = list(net_info.keys())
        
        #get all the unique rec layers in the multiexperiment
        all_rec_layers = [(conf['subject']['rec_layers']).split(',') for conf in self._search_config]
        unique_rec_layers = list({string for sublist in all_rec_layers for string in sublist})
        
        for rl in unique_rec_layers:
            if 'r' in rl: # If random units are recorded (NOTE: FOR NOW IN THIS IF WE EXPECT ONLY CONV LAYERS)
                #get the shape of the layer of interest
                layer_idx , units = rl.split('=')
                shape      = net_info[layer_names[int(layer_idx)]][1:]
                n_rand, _ = units.split('r'); n_rand = int(n_rand)
                # Unravel indexes sampled in the overall interval
                neurons = np.unravel_index(
                    indices=np.random.choice(
                        a = np.prod(shape),
                        size=n_rand,
                        replace=False,
                    ),
                    shape=shape)
                #convert the sampled neurons into string format
                neurons_str ='='.join([layer_idx, str([tuple([neurons[c_i][i] for c_i in range(len(neurons))]) 
                                                            for i in range(len(neurons[0]))] ).replace(',', '') ]) 
                for config in self._search_config:
                    config['subject']['rec_layers'] = config['subject']['rec_layers'].replace(rl, neurons_str)
        
        #get all the unique score layers in the multiexperiment
        all_score_layers = [(conf['scorer']['scr_layers']).split(',') for conf in self._search_config]
        unique_score_layers = list({layer_names[int(string.split('=')[0])] for sublist in all_score_layers for string in sublist})
        scored_units_dict = {k:[] for k in unique_score_layers}
        
        for config in self._search_config:
        
            #NOTE: we assume that in maximize activity multiexp you score from
            #one layer per experiment
            layer_idx , units = config['scorer']['scr_layers'].split('=')
            n_rand, _ = units.split('r'); n_rand = int(n_rand)
            
            for s in config['subject']['rec_layers'].split(','):
                l_nr, rec_units = s.split('=')
                if int(l_nr) == int(layer_idx):
                    break
            if rec_units.strip()=='[]':
                #case 1) all units recorded    
                n_rec = net_info[layer_names[int(layer_idx)]][1:][-1]
            elif '(' in rec_units:
                #case 2) convolutional layer -> NEVER all units recorded
                n_rec = len(rec_units.split(') ('))
            else:
                #case 2) linear layer, only some units recorded
                n_rec = len(rec_units.split(' '))
            
            
            idx_to_score = list(
                np.random.choice(
                    a=np.arange(0,n_rec), 
                    size=n_rand, 
                    replace=False
                )
            )
            
            if rec_units.strip()=='[]':
                scored_units = idx_to_score
            else:
                splitter = ') (' if '(' in rec_units else ' '
                scored_units = [rec_units.strip('[]').split(splitter)[idx] for idx in idx_to_score]
                if '(' in rec_units:
                    for i,s in enumerate(scored_units):
                        if not s.startswith('('):
                            scored_units[i] = '('+ scored_units[i]
                        if not s.endswith('('):
                            scored_units[i] = scored_units[i]+ ')'
                            
            config['scorer']['scr_layers'] = '='.join([layer_idx,str(scored_units).replace("'", "").replace(",", "")])
            scored_units_dict[layer_names[int(layer_idx)]].append(config['scorer']['scr_layers'].split('=')[1])
"""




