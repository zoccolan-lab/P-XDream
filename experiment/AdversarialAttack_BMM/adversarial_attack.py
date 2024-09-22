from os import path
from typing import Any, Dict, Iterable, List, Tuple, cast
from einops import rearrange, reduce

import numpy  as np
import torch
from numpy.typing import NDArray
from PIL          import Image
from torchvision.transforms.functional import to_pil_image

from pxdream.experiment                import ZdreamExperiment
from pxdream.generator                 import Generator, DeePSiMGenerator
from pxdream.optimizer                 import GeneticOptimizer, Optimizer
from pxdream.scorer                    import ActivityScorer, Scorer, ParetoReferencePairDistanceScorer
from pxdream.subject                   import InSilicoSubject, TorchNetworkSubject
from pxdream.utils.dataset             import ExperimentDataset, MiniImageNet, NaturalStimuliLoader
from pxdream.utils.logger              import DisplayScreen, Logger, LoguruLogger
from pxdream.utils.message             import ZdreamMessage
from pxdream.utils.misc                import concatenate_images, device
from pxdream.utils.parameters          import ArgParams, ParamConfig
from pxdream.utils.probe               import RecordingProbe
from pxdream.utils.types               import Codes, ScoringUnits, Stimuli, Fitness, States
from experiment.MaximizeActivity.plot import multiexp_lineplot, plot_optimizing_units, plot_scores, plot_scores_by_label, save_best_stimulus_per_variant, save_stimuli_samples
from experiment.utils.args            import ExperimentArgParams
from experiment.utils.parsing         import parse_boolean_string, parse_recording, parse_scoring, parse_signature
from experiment.utils.misc            import BaseZdreamMultiExperiment, make_dir

class AdversarialAttackExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "AdversarialAttack"

    # Screen titles for the display
    NAT_IMG_SCREEN = 'Synthetic 1'
    GEN_IMG_SCREEN = 'Synthetic 2'
    
    # --- OVERRIDES ---
    
    # We override the properties to cast them to the specific components to activate their methods

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer,      self._scorer ) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 
    
    # --- CONFIG ---

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'AdversarialAttackExperiment':
        '''
        Create a AdversarialAttackExperiment instance from a configuration dictionary.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: AdversarialAttackExperiment instance with hyperparameters set from configuration.
        :rtype: AdversarialAttack
        '''
        
        # Extract parameter from configuration and cast
        
        PARAM_weights    = str  (conf[ExperimentArgParams.GenWeights     .value])
        PARAM_variant    = str  (conf[ExperimentArgParams.GenVariant     .value])
        PARAM_template   = str  (conf[ExperimentArgParams.Template       .value])
        PARAM_dataset    = str  (conf[ExperimentArgParams.Dataset        .value])
        PARAM_shuffle    = bool (conf[ExperimentArgParams.Shuffle        .value])
        PARAM_batch      = int  (conf[ExperimentArgParams.BatchSize      .value])
        PARAM_net_name   = str  (conf[ExperimentArgParams.NetworkName    .value])
        PARAM_rec_layers = str  (conf[ExperimentArgParams.RecordingLayers.value])
        PARAM_scr_layers = str  (conf[ExperimentArgParams.ScoringLayers  .value])
        PARAM_scr_sign   = str  (conf[ExperimentArgParams.ScoringSignature.value])
        PARAM_distance   = str  (conf[ExperimentArgParams.Distance       .value])
        PARAM_unit_red   = str  (conf[ExperimentArgParams.UnitsReduction .value])
        PARAM_layer_red  = str  (conf[ExperimentArgParams.LayerReduction .value])
        PARAM_pop_size   = int  (conf[ExperimentArgParams.PopulationSize .value])
        PARAM_sigma0     = float(conf[ExperimentArgParams.Sigma0         .value])
        PARAM_exp_name   = str  (conf[          ArgParams.ExperimentName .value])
        PARAM_iter       = int  (conf[          ArgParams.NumIterations  .value])
        PARAM_rnd_seed   = int  (conf[          ArgParams.RandomSeed     .value])
        PARAM_render     = bool (conf[          ArgParams.Render         .value])
        
        PARAM_close_screen = conf.get(ArgParams.CloseScreen.value, True)
        
        # Set numpy random seed
        
        np.random.seed(PARAM_rnd_seed)

        # --- NATURAL IMAGE LOADER ---

        # Parse template and derive if to use natural images
        template = parse_boolean_string(boolean_str=PARAM_template) 
        use_nat  = template.count(False) > 0
        
        # Create dataset and loader
        dataset  = MiniImageNet(root=PARAM_dataset)
        
        nat_img_loader   = NaturalStimuliLoader(
            dataset=dataset,
            template=template,
            shuffle=PARAM_shuffle,
            batch_size=PARAM_batch
        )
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root    = str(PARAM_weights),
            variant = str(PARAM_variant) # type: ignore
        ).to(device)
        
        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(
            network_name=str(PARAM_net_name)
        ).layer_info

        # Probe
        record_target = parse_recording(input_str=PARAM_rec_layers, net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name,
        )
        
        # Set the network in evaluation mode
        sbj_net.eval()

        # --- SCORER ---

        # Parse target neurons
        scoring_units = parse_scoring(
            input_str=str(PARAM_scr_layers), 
            net_info=layer_info,
            rec_info=record_target
        )

        signature = parse_signature(
            input_str=str(PARAM_scr_sign),
            net_info=layer_info,
        )

        scorer = ParetoReferencePairDistanceScorer(
            layer_weights=signature,
            scoring_units=scoring_units,
            metric=str(PARAM_distance),
        )

        # --- OPTIMIZER ---

        # optim = CMAESOptimizer(
        #     codes_shape = generator.input_dim,
        #     rnd_seed    = PARAM_rnd_seed,
        #     pop_size    = PARAM_pop_size,
        #     sigma0      = PARAM_sigma0
        # )
        
        optim = GeneticOptimizer(
            codes_shape  = (2, *generator.input_dim),
            rnd_seed     = PARAM_rnd_seed,
            pop_size     = PARAM_pop_size,
            rnd_scale    = 1,
            mut_size     = 0.3, #0.6
            mut_rate     = 0.3, #0.25
            allow_clones = True,
            n_parents    = 4
        )

        #  --- LOGGER --- 

        conf[ArgParams.ExperimentTitle.value] = AdversarialAttackExperiment.EXPERIMENT_TITLE
        # logger = LoguruLogger(path=Logger.path_from_conf(conf=conf)) NOT IN ISCHIAGUALASTIA BABY :)
        logger = LoguruLogger(on_file=False)
        
        # In the case render option is enabled we add display screens
        if bool(PARAM_render):

            # In the case of multi-experiment run, the shared screens
            # are set in `Args.DisplayScreens` entry
            if ArgParams.DisplayScreens.value in conf:
                for screen in cast(Iterable, conf[ArgParams.DisplayScreens.value]):
                    logger.add_screen(screen=screen)

            # If the key is not set it is the case of a single experiment
            # and we create screens instance
            else:

                # Add screen for synthetic images
                logger.add_screen(
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE)
                )

                # Add screen fro natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE)
                    )

        # --- DATA ---
        
        data = {
            "dataset"      : dataset if use_nat else None,
            'render'       : PARAM_render,
            'close_screen' : PARAM_close_screen,
            'use_nat'      : use_nat,
            'n_group'      : 2
        }

        # --- EXPERIMENT INSTANCE ---
        
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            nat_img_loader = nat_img_loader,
            iteration      = PARAM_iter,
            data           = data, 
            name           = PARAM_exp_name
        )

        return experiment
    
    # --- INITIALIZATION ---

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
        name           : str            = 'adversarial_attack'
    ) -> None:
        ''' 
        Uses the same signature as the parent class ZdreamExperiment.
        It save additional information from the `data` attribute to be used during the experiment.
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
        self._render        = cast(bool, data[str(ArgParams.Render)])
        self._close_screen  = cast(bool, data['close_screen'])
        self._use_natural   = cast(bool, data['use_nat'])
        self._n_group       = cast(int,  data['n_group'])
        
        # Save dataset if used
        if self._use_natural: self._dataset = cast(ExperimentDataset, data['dataset'])

                
    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.best_code
        
        best_code = rearrange(best_code, f'b g ... -> (b g) ...', g=self._n_group)
        best_synthetic = [self.generator(codes=msg.best_code[:,c_id,:]) for c_id in range(msg.best_code.shape[1])]
        best_synthetic_img = concatenate_images(best_synthetic[0])

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
    # --- DATA FLOW ---
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        Override the method to save the last set of stimuli and the labels if natural images are used.
        '''

        # We save the last set of stimuli in `self._stimuli`
        self._stimuli, msg = data
        
        # We save the last set of labels if natural images are used
        if self._use_natural: self._labels.extend(msg.labels)
        
        states, msg = super()._stimuli_to_states(data)

        return states, msg

    def _scores_to_codes(self, data: Tuple[Fitness, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        ''' We use scores to save stimuli (both natural and synthetic) that achieved the highest score. '''

        sub_score, msg = data
        m = reduce(msg.mask, '(b g) ... -> b ...', 'all', g=self._n_group)
        # We inspect if the new set of stimuli for natural images,
        # checking if they achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]
                
        scores_ = sub_score[m]
        
        # Optimizer step
        codes = self.optimizer.step(scores=scores_)
    
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg

    def _states_to_scores(self, data: Tuple[States, ZdreamMessage]) -> Tuple[Fitness, ZdreamMessage]:
        '''
        The method evaluate the SubjectResponse in light of a Scorer logic.

        :param data: Subject responses to visual stimuli and a Message
        :type data: Tuple[State, Message]
        :return: A score for each presented stimulus.
        :rtype: Tuple[Score, Message]
        '''
        
        states, msg = data
        
        # Here we sort synthetic from natural images such that grouping works as expected
        # sort_m = np.concatenate([np.nonzero(msg.mask), np.nonzero(~msg.mask)])
        states_ = {k : rearrange(v, '(b g) ... -> b g ...', g=self._n_group) for k, v in states.items()}
        
        # Here we sort the mask array in descending order so that all the True come in front
        # and is then easier to just disregard later the natural images scores
        # msg.mask[::-1].sort()
        
        scores = self.scorer.__call__(states=states_)
        
        # Return mask to appropriate dimension
        m = reduce(msg.mask, '(b g) ... -> b ...', 'all', g=self._n_group)
    
        msg.scores_gen_history.append(scores[ m])
        msg.scores_nat_history.append(scores[~m])
        
        return scores, msg
    
    def _codes_to_stimuli(self, data: Tuple[Codes, ZdreamMessage]) -> Tuple[Stimuli, ZdreamMessage]:
        
        codes, msg = data
        
        # In the adversarial attack experiment each code represent
        # a pair of images, so is expected to have double the size
        # required by the generator, we override this hook to properly
        # resize the codes to split them in half and stack them along
        # the batch dimension
        codes_ = rearrange(codes, f'b g ... -> (b g) ...', g=self._n_group)
    
        data_ = (codes_, msg)
        
        return super()._codes_to_stimuli(data=data_)
    # --- RUN METHODS ---

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:
        ''' Add information about the best score and the average score.'''
        
        # Best synthetic scores
        stat_gen = msg.stats_gen
        
        best_gen     = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen     = cast(NDArray, stat_gen['curr_score']).mean()
        
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        layerwise_score = " ".join([f'{k}:{np.mean(v):.1f}' for k,v in self._scorer.layer_scores.items()])
         
        
        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str} | unweighted scores: {layerwise_score}'
        
        # Best natural score
        if self._use_natural:
            
            stat_nat     = msg.stats_nat
            best_nat     = cast(NDArray, stat_nat['best_score']).mean()
            best_nat_str = f'{best_nat:.1f}'
            
            desc = f'{desc} | {best_nat_str}'

        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super} {desc}'

    def _init(self) -> ZdreamMessage:
        ''' We initialize the experiment and add the data structure to save the best score and best image. '''

        msg = super()._init()

        # Data structures to save best natural images and labels presented
        # Note that we don't keep track for the synthetic image as it is already done by the optimizer
        if self._use_natural:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
            self._labels: List[int] = []
        
        # Gif
        self._gif: List[Image.Image] = []

        return msg


    def _finish(self, msg : ZdreamMessage):
        ''' 
        Save best stimuli and make plots
        '''

        msg = super()._finish(msg)
        
        
    
        # DO YOU WANT TO SAVE ALL THE SHIT BELOW? NOT IN ISCHIAGUALASTIA, LET'S SAVE SOME SPACE

        # Close screens if set (or preserve for further experiments using the same screen)
        if self._close_screen: self._logger.close_all_screens()
            
        # 1. SAVE VISUAL STIMULI (SYNTHETIC AND NATURAL)

        # Create image folder
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = [self.generator(codes=msg.best_code[:,c_id,:])[0] for c_id in range(msg.best_code.shape[1])]

        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(bg), f'adv_best{i}') for i,bg in enumerate(best_gen)]
        
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
            
        return msg
        # Saving gif
        # out_fp = path.join(img_dir, 'evolving_best.gif')
        # self._logger.info(f'> Saving evolving best stimuli across generations to {out_fp}')
        # to_gif(image_list=self._gif, out_fp=out_fp)

        self._logger.info(mess='')
        
        # 2. SAVE PLOTS

        # Create plot folder
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

        # Plot scores by label
        if self._use_natural:
            
            plot_scores_by_label(
                scores=(
                    np.stack(msg.scores_gen_history),
                    np.stack(msg.scores_nat_history)
                ),
                lbls=self._labels,
                out_dir=plots_dir, 
                dataset=self._dataset,
                logger=self._logger
            )
        
        self._logger.reset_formatting()
        self._logger.info(mess='')
        
        return msg