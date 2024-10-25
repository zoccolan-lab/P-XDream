from collections import defaultdict
import glob
from os import path
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, cast
import pandas as pd

import numpy  as np
import torch
from numpy.typing import NDArray
from PIL          import Image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from scipy.spatial.distance import pdist

from pxdream.experiment                import ParetoExperimentState, ZdreamExperiment
from pxdream.generator                 import Generator, DeePSiMGenerator
from pxdream.optimizer                 import CMAESOptimizer, GeneticOptimizer, Optimizer
from pxdream.scorer                    import Scorer, ParetoReferencePairDistanceScorer, _MetricKind
from pxdream.subject                   import InSilicoSubject, TorchNetworkSubject
from pxdream.utils.dataset             import ExperimentDataset, MiniImageNet, NaturalStimuliLoader, RandomImageDataset
from pxdream.utils.io_ import load_pickle
from pxdream.utils.logger              import DisplayScreen, Logger, LoguruLogger
from pxdream.utils.message             import ParetoMessage, ZdreamMessage
from pxdream.utils.misc                import concatenate_images, device
from pxdream.utils.parameters          import ArgParams, ParamConfig
from pxdream.utils.probe               import RecordingProbe
from pxdream.utils.types               import Codes, Stimuli, Fitness, States
from experiment.utils.args            import ExperimentArgParams
from experiment.utils.parsing         import parse_boolean_string, parse_bounds, parse_net_loading, parse_recording, parse_reference_info, parse_scoring, parse_signature
from experiment.utils.misc            import BaseZdreamMultiExperiment, make_dir

class StretchSqueezeMaskExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "AdversarialAttackMax"

    # Screen titles for the display
    NAT_IMG_SCREEN = 'Reference'
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # --- OVERRIDES ---
    
    # We override the properties to cast them to the specific components to activate their methods

    @property
    def scorer(self)  -> ParetoReferencePairDistanceScorer:      return cast(ParetoReferencePairDistanceScorer,      self._scorer ) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 
    
    # --- CONFIG ---

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'StretchSqueezeMaskExperiment':
        '''
        Create a MaximizeActivityExperiment instance from a configuration dictionary.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: MaximizeActivity
        '''
        
        # Extract parameter from configuration and cast
        
        PARAM_weights     = str  (conf[ExperimentArgParams.GenWeights      .value])
        PARAM_variant     = str  (conf[ExperimentArgParams.GenVariant      .value])
        PARAM_template    = str  (conf[ExperimentArgParams.Template        .value])
        PARAM_dataset     = str  (conf[ExperimentArgParams.Dataset         .value])
        PARAM_shuffle     = bool (conf[ExperimentArgParams.Shuffle         .value])
        PARAM_batch       = int  (conf[ExperimentArgParams.BatchSize       .value])
        PARAM_net_name    = str  (conf[ExperimentArgParams.NetworkName     .value])
        PARAM_rec_layers  = str  (conf[ExperimentArgParams.RecordingLayers .value])
        PARAM_scr_layers  = str  (conf[ExperimentArgParams.ScoringLayers   .value])
        PARAM_scr_sign    = str  (conf[ExperimentArgParams.ScoringSignature.value])
        PARAM_bounds      = str  (conf[ExperimentArgParams.Bounds          .value])
        PARAM_customW_path = str  (conf[ExperimentArgParams.CustomWeightsPath     .value])
        PARAM_customW_var  = str  (conf[ExperimentArgParams.CustomWeightsVariant  .value])
        PARAM_distance    = str  (conf[ExperimentArgParams.Distance        .value])
        PARAM_pop_size    = int  (conf[ExperimentArgParams.PopulationSize  .value])
        PARAM_exp_name    = str  (conf[          ArgParams.ExperimentName  .value])
        PARAM_iter        = int  (conf[          ArgParams.NumIterations   .value])
        PARAM_rnd_seed    = int  (conf[          ArgParams.RandomSeed      .value])
        PARAM_render      = bool (conf[          ArgParams.Render          .value])
        PARAM_ref         = str  (conf[ExperimentArgParams.Reference       .value])
        PARAM_ref_info    = str  (conf[ExperimentArgParams.ReferenceInfo   .value])
        PARAM_sigma0      = float(conf[ExperimentArgParams.Sigma0          .value])
        PARAM_optim_type  = str  (conf[ExperimentArgParams.OptimType       .value])
        PARAM_net_loading = str  (conf[ExperimentArgParams.WeightLoadFunction.value])

        PARAM_close_screen = conf.get(ArgParams.CloseScreen.value, True)
        path2CustomW = os.path.join(PARAM_customW_path, PARAM_net_name, PARAM_customW_var) if PARAM_customW_var else ''
        
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

        #Get net loading function from parsing
        net_loading = parse_net_loading(input_str = PARAM_net_loading)
        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name,
            t_net_loading = net_loading,
            custom_weights_path = path2CustomW
        )
        
        # Set the network in evaluation mode
        sbj_net.eval()
        # --- SCORER ---
        
        reference_file      = load_pickle(PARAM_ref)
        gen_var, layer, neuron, seed = parse_reference_info(PARAM_ref_info)
        ref_info = {'gen_var': gen_var, 'layer': layer, 'neuron': neuron, 'seed': seed,
                    'ref_file': PARAM_ref}
        
        layer_name = list(layer_info.keys())[layer]
        
        # Extract code from reference file
        r = '_r' if path2CustomW else ''
        net_key = PARAM_net_name+r
        
        try:
            ref_code = reference_file['reference'][net_key][gen_var][layer_name][neuron][seed]['code']
        except KeyError:
            raise ValueError(f'No reference found for gen_variant {gen_var}, layer {layer_name}, neuron {neuron}, seed {seed} in file {PARAM_ref}')
        
        # Generate the code and the state, unbatching it
        ref_stimulus : Stimuli = generator(codes=ref_code)
        ref_states_b : States  = sbj_net(stimuli=ref_stimulus)
        print(f"Reference performance: {ref_states_b[list(record_target.keys())[-1]]}")
        
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
        
        bounds = parse_bounds(
            input_str=str(PARAM_bounds),
            net_info=layer_info,
            reference=ref_states_b                
        )
        
        sel_metric = str(PARAM_distance)
        sel_metric = cast(_MetricKind, sel_metric)
        
        # Generate reference        
        scorer = ParetoReferencePairDistanceScorer(
            layer_weights=signature,
            scoring_units=scoring_units,
            reference = (ref_states_b,ref_info),
            metric=sel_metric,
            bounds = bounds
        )

        # --- OPTIMIZER ---
        if PARAM_optim_type == 'cmaes':
            optim = CMAESOptimizer(
                codes_shape = generator.input_dim,
                rnd_seed    = PARAM_rnd_seed,
                pop_size    = PARAM_pop_size,
                sigma0      = PARAM_sigma0,
                x0          = ref_code
            )
        elif PARAM_optim_type == 'genetic':
            optim = GeneticOptimizer(
                codes_shape  = generator.input_dim,
                rnd_seed     = PARAM_rnd_seed,
                pop_size     = PARAM_pop_size,
                rnd_scale    = 1,
                mut_size     = 0.3,
                mut_rate     = 0.3,
                allow_clones = True,
                n_parents    = 4
            )
        else:
            raise ValueError(f'Optimizer type {PARAM_optim_type} not recognized')
            

        #  --- LOGGER --- 

        conf[ArgParams.ExperimentTitle.value] = StretchSqueezeMaskExperiment.EXPERIMENT_TITLE
        #logger = LoguruLogger.from_conf(conf=conf)
        logger = LoguruLogger(to_file=False)
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
            'ref_code'     : ref_code
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
            name           = PARAM_exp_name,
            
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

        self._readout_score = defaultdict(list)

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
        self._reference_code = cast(NDArray, data['ref_code'])
        self._render         = cast(bool,    data[str(ArgParams.Render)])
        self._close_screen   = cast(bool,    data['close_screen'])
        self._use_natural    = cast(bool,    data['use_nat'])
        
        # Save dataset if used
        if self._use_natural: self._dataset = cast(ExperimentDataset, data['dataset'])  
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

    def _states_to_scores(self, data: Tuple[States, ParetoMessage]) -> Tuple[Fitness, ParetoMessage]:
        '''
        The method evaluate the SubjectResponse in light of a Scorer logic.

        :param data: Subject responses to visual stimuli and a Message
        :type data: Tuple[State, Message]
        :return: A score for each presented stimulus.
        :rtype: Tuple[Score, Message]
        '''
        
        states, msg = data
        
        # Here we sort the mask array in descending order so that all the True come in front
        # and is then easier to just disregard later the natural images scores
        # msg.mask[::-1].sort()
        
        states, msg = data
        
        # Scorer step
        _, self.layer_scores, scores = self.scorer(states=states)
        
        if scores is not None:
            # Update message scores history (synthetic and natural)
            msg.scores_gen_history.append(scores[ msg.mask])
            msg.scores_nat_history.append(scores[~msg.mask])
                        
        for k,v in self.layer_scores.items():
            msg.layer_scores_gen_history[k].append(v) #[ msg.mask]?
        
        msg.local_p1.append(np.column_stack((np.full(self.scorer.coordinates_p1.shape[0], 
                                            self._curr_iter), self.scorer.coordinates_p1)) )
        msg.scores_gen_history.append(scores[ msg.mask])
        msg.scores_nat_history.append(scores[~msg.mask])
        
        return scores, msg
    
    def _scores_to_codes(self, data: Tuple[Fitness, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        ''' We use scores to save stimuli (both natural and synthetic) that achieved the highest score. '''

        sub_score, msg = data

        # We inspect if the new set of stimuli for natural images,
        # checking if they achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._scores_to_codes((sub_score, msg))
    
    def _codes_to_stimuli(self, data: Tuple[Codes, ZdreamMessage]) -> Tuple[Stimuli, ZdreamMessage]:
        
        codes, msg = data
        
        # Aggiungi il codice a quello della reference per generare
        codes_ = codes + self._reference_code
    
        data_ = (codes_, msg)
        
        return super()._codes_to_stimuli(data=data_)
    
    def _init(self) -> ParetoMessage:
        ''' We initialize the experiment and add the data structure to save the best score and best image. '''
        msg = super(ZdreamExperiment, self)._init()
        msg = ParetoMessage(
            start_time = msg.start_time,
            rec_units  = self.subject.target,
            scr_units  = self.scorer.scoring_units,
            signature  = self.scorer._layer_weights,
        )
        # Data structures to save best natural images and labels presented
        # Note that we don't keep track for the synthetic image as it is already done by the optimizer
        if self._use_natural:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
            self._labels: List[int] = []
        
        # Gif
        self._gif: List[Image.Image] = []

        return msg
    
    def _progress(self, i: int, msg : ParetoMessage):

        super()._progress(i, msg)
        
        # Best synthetic stimulus (and natural one)
        best_synthetic     = self.generator(codes=msg.best_code)
        best_synthetic_img = to_pil_image(best_synthetic[0])
                
        if self._use_natural: best_natural = self._best_nat_img
        
        # Update gif if different from the previous frame
        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )          
        
        # Update screens
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
                
        # Check early stopping
        self.pareto_early_stopping(msg, valid_val_thrshold = 10, iter_thrshold = 5)
    
    def _progress_info(self, i: int, msg : ParetoMessage) -> str:
        ''' Add information about the best score and the average score.'''
        
        # TODO: UPDATE WITH PARETO STATS
        # Best synthetic scores
        
        layerwise_score = " ".join([f'{k}:{np.mean(v):.1f}' for k,v in self.layer_scores.items()])
        #NOTE: layerwise score for input is the pixel budjet
        
        desc = f'Unweighted scores: {layerwise_score}'
        
        # Best natural score
        if self._use_natural:
            stat_nat     = msg.stats_nat
            best_nat     = cast(NDArray, stat_nat['best_score']).mean()
            best_nat_str = f'{best_nat:.1f}'
            
            desc = f'{desc} | {best_nat_str}'

        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super} {desc}'
    
    def _finish(self, msg : ParetoMessage):
        ''' 
        Save best stimuli and make plots
        '''
        msg.get_pareto1()
        msg = super(ZdreamExperiment, self)._finish(msg = msg) # type: ignore

        # Dump
        state = ParetoExperimentState.from_msg(msg=msg)
        state.dump(
            out_dir=path.join(self.dir, 'state'),
            logger=self._logger)
         
        # Close screens if set (or preserve for further experiments using the same screen)
        if self._close_screen: self._logger.close_all_screens()
            
        # 1. SAVE VISUAL STIMULI (SYNTHETIC AND NATURAL)

        # Create image folder
        img_dir_gen = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)
        df_row = self.save_exp_data(img_dir_gen); self.img_dir = df_row['image_path']
        self.best_syn = self.generator(codes=(self._reference_code))[0]
        #self.best_gen = self.generator(codes=msg.best_code+self._reference_code)[0]
        self.best_gen = self.generator(codes=msg.best_code)[0]
        #self.best_gen = self.generator(codes=state.codes[msg.best_code_idx[0]+1:,msg.best_code_idx[1],:])[0]
        self._save_images(img_dir=self.img_dir, best_gen=self.best_gen, ref=self.best_syn)
        
        last_layer = list(state.layer_scores_gen_history.keys())[-1] #type: ignore
        input_layer = list(state.layer_scores_gen_history.keys())[0] #type: ignore
        
        # Save variables to log in a multi-experiment csv
        self.hidden_reference = int(self.scorer._reference[last_layer])
        self.distance         = get_best_distance(self.img_dir)
        self.hidden_dist      = state.layer_scores_gen_history[last_layer][msg.best_code_idx[0],msg.best_code_idx[1]] #type: ignore
        self.image_dist       = state.layer_scores_gen_history[input_layer][msg.best_code_idx[0],msg.best_code_idx[1]] #type: ignore
        print(f"image dist - rec {self.image_dist}, direct computation {self.distance}")
        #exp_path = os.path.join(img_dir_gen, 'experiments.csv')
        #df = pd.read_csv(exp_path) if os.path.exists(exp_path) else pd.DataFrame(columns=[*df_row.keys(),'experiment', 
        #                                                                                'hidden_reference', 'hidden_dist', 'pixel_dist', 'num_iter'])
        
        # df_row['hidden_reference']= int(self.hidden_reference)
        # df_row['hidden_dist'] = hidden_dist
        # df_row['pixel_dist'] = distance
        # df_row['num_iter'] = self._curr_iter
        # 
        # df = pd.concat([df, pd.DataFrame([df_row])])
        # df.to_csv(exp_path, index=False)
        
        return msg
        
    def save_exp_data(self, img_dir_gen):
        # usato per salvare immagini e dati dell'esperimento. Va messo a posto
        target_cat = int(self.subject.target[list(self.subject.target.keys())[-1]][0][0]) # non capito il type checker
        task = 'invar' if list(self.scorer._layer_weights.values())[0] < 0 else 'adv_attk' #rough
        robust = '_r' if self.subject.robust else ''
        net_name = f"{self.subject._name}{robust}"; low_layer = list(self.scorer._layer_weights.keys())[0]
        category = f"c{target_cat}"
        dirname = f"{task}_{net_name}_{low_layer}_{category}"
        new_base_folder = os.path.join(img_dir_gen, self._name)
        new_task_folder = os.path.join(new_base_folder, f"{task}_{net_name}")
        new_layer_folder = os.path.join(new_task_folder, low_layer)
        new_category_folder = os.path.join(new_layer_folder, category)
        os.makedirs(new_category_folder, exist_ok=True)
        subdirs = [name for name in os.listdir(new_category_folder) if os.path.isdir(os.path.join(new_category_folder, name)) and re.search(dirname, name)]
        dirname = f"{dirname}_{len(subdirs) + 1}_rs{self._optimizer._rnd_seed}"        
        img_dir = make_dir(path=path.join(new_category_folder, dirname), logger=self._logger)
        df_row = {'multiexp':self._name,
                'task':task,
                'network':net_name,
                'low_layer':low_layer,
                'category':target_cat,
                'seed':self._optimizer._rnd_seed,
                'image_path': img_dir}

        return df_row
    
    def _save_images(self, img_dir: str, best_gen: torch.Tensor, ref: torch.Tensor):  
        
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen), 'best adversarial')]
        to_save.append((to_pil_image(ref), 'best synthetic'))
        to_save.append((concatenate_images(img_list=[ref, best_gen]), 'comparison'))
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
            
    def pareto_early_stopping(self, msg : ParetoMessage, valid_val_thrshold = 10, iter_thrshold = 5):
        '''
        Determines whether to trigger early stopping based on Pareto front constraints.

        This method evaluates the scores of the last layer and checks if the number of valid
        values (i.e., values not equal to negative infinity) is below a certain threshold.
        If the number of valid values is less than `valid_val_thrshold` and the current iteration count exceeds `iter_thrshold`,
        early stopping is triggered by setting the `early_stopping` attribute of the `msg` object to True.

        :param msg: An object containing the early stopping flag.
        :type msg: ParetoMessage
        :param valid_val_thrshold: The threshold for the number of valid values to trigger early stopping.
        :type valid_val_thrshold: int
        :param iter_thrshold: The iteration threshold to trigger early stopping.
        :type iter_thrshold: int
        :return: None
        :rtype: None
        '''
        # TODO: IMPROVE AND GENERALIZE

        last_layer = list(self.layer_scores.keys())[-1]
        #apply constrains to layer scores
        bounded_lscores = self.scorer.bound_constraints(state = self.layer_scores)
        #count the number of valid values (i.e. not equal to -inf)
        valid_values_count = sum(1 for value in bounded_lscores[last_layer] 
            if value != float('-inf')
        )
        if valid_values_count < valid_val_thrshold and self._curr_iter > iter_thrshold:
                msg.early_stopping = True

class StretchSqueezeExperiment(StretchSqueezeMaskExperiment):
    
    def _run_init(self, msg : ParetoMessage) -> Tuple[Codes, ParetoMessage]:
        '''
        Method called before entering the main for-loop across generations.
        It is responsible for generating the initial codes
        '''
    
        # Codes initialization
        if isinstance(self.optimizer, CMAESOptimizer):
            codes = self.optimizer.init()
        else:
            ref_codes = np.tile(self._reference_code, (self._optimizer._init_n_codes, 1))
            noise = np.random.normal(0, np.sqrt(0.01 * np.abs(np.mean(self._reference_code))), ref_codes.shape)
            init_codes = (ref_codes + noise)
            codes = self.optimizer.init(init_codes = init_codes)
        
        # Update the message codes history
        msg.codes_history.append(codes)
        
        return codes, msg
    
    def _codes_to_stimuli(self, data: Tuple[Codes, ZdreamMessage]) -> Tuple[Stimuli, ZdreamMessage]:
        
        codes, msg = data
        
        codes_ = codes
    
        data_ = (codes_, msg)
        
        return super(StretchSqueezeMaskExperiment, self)._codes_to_stimuli(data=data_)
    
    
    def _finish(self, msg : ParetoMessage):
        ''' 
        Save best stimuli and make plots.
        
        '''
        
        msg = super()._finish(msg = msg)
        #self.best_gen = self.generator(codes=msg.best_code)[0]
        self.best_gen = self.generator(msg.codes_history[-1][msg.best_code_idx[1],:].reshape(1, 4096))[0]
        self._save_images(img_dir=self.img_dir, best_gen=self.best_gen, ref=self.best_syn)
        self.distance         = get_best_distance(self.img_dir)
        print(self.distance)
        return msg
    
    
#TODO: check why this is different from the recorded image. Is it because of the normalization?    
def get_best_distance(im_dir):
    #per calcolare empiricamente la distanza euclidea tra due immagini.
    #usato per adversarial attacks project @BMM
    def load_myimg(path):
        img = Image.open(path)
        transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        ])
        image_tensor = torch.unsqueeze(transform(img), dim = 0).to('cuda')
        #torch.nn.functional.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return image_tensor
    
    adv_images = {os.path.basename(imp):load_myimg(imp) 
                    for imp in glob.glob(os.path.join(im_dir, '*.png')) 
                    if any(s in os.path.basename(imp) for s in ['adversarial', 'synthetic'])}
    
    matrix = np.vstack([t.view(-1).cpu().numpy() for k,t in adv_images.items()])
    # Calcolare la distanza euclidea tra i due vettori
    distance = pdist(matrix, 'euclidean')[0]
    print(f"Distanza euclidea: {distance}")
    return distance

#TODO: refactor anche l'esperimento AdvAttk originale (evolverne 2 insieme)
#??? Classe ParetoExperiment ???
#class AdversarialAttackExperimentEvolvePair(StretchSqueezeMaskExperiment):
    
    

class StretchSqueezeLayerMultiExperiment(BaseZdreamMultiExperiment):
    
    def _init(self):
        
        super()._init()
        
        self._data['desc'] = 'StretchSqueeze Experiment Targeting multiple Layers in the same Network'
        
        self._data['net_sbj']        = []
        self._data['robust']         = []
        self._data['lower_ly']       = []
        self._data['upper_ly']       = []
        self._data['low_target']     = []
        self._data['high_target']    = []
        self._data['task_signature'] = []
        self._data['rnd_seed']       = []
        self._data['reference_info'] = []
        self._data['reference_activ']= []
        self._data['p1_front']       = []
        self._data['p1_codes']       = []
        self._data['layer_scores']   = []
        self._data['num_iter']       = []

    def _progress(
        self, 
        exp  : StretchSqueezeExperiment, 
        conf : ParamConfig,
        msg  : ParetoMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        # get the network type (robust or not)
        self._data['net_sbj'].append(exp.subject._name)
        self._data['robust'].append(exp.subject.robust)
        # Collect the needed data from the experiment
        #get the layers involved and their target neurons
        low_key, high_key = list(exp.subject.target.keys()) 
        self._data['lower_ly'].append(low_key)
        self._data['upper_ly'].append(high_key)
        self._data['low_target'].append(exp.subject.target[low_key])
        self._data['high_target'].append(exp.subject.target[high_key])
        #get the signature, which is characteristic of the task (invariance or adversarial attack)
        self._data['task_signature'].append(exp.scorer._layer_weights)
        self._data['rnd_seed'].append(exp.optimizer._rnd_seed)
        self._data['reference_info'].append(exp.scorer.reference_info)
        self._data['reference_activ'].append(exp.scorer._reference[high_key])
        pf1_coords = msg.Pfront_1
        self._data['p1_front'].append(pf1_coords)
        codes_history = np.stack(msg.codes_history)
        #TODO: why codes_history length is nr iter +1?
        self._data['p1_codes'].append(codes_history[pf1_coords[:,0]+1, pf1_coords[:,1],:]),
        self._data['layer_scores'].append({k:np.vstack(v) for k,v in msg.layer_scores_gen_history.items()})
        self._data['num_iter'].append(exp._curr_iter)
        

    def _finish(self):
        self.get_df_summary()
        #TODO: summary of the experiment as .xlsx file?
        super()._finish() 
        
    def get_df_summary(self):
        Sign2Task = {
            1 : 'adversarial',
            -1 : 'invariance'
        }
        mexp_data = self._data
        # Funzione per verificare se tutti gli elementi di una lista sono di un tipo desiderato
        def is_valid_column(column):
            valid_types = (str, int, float, bool)
            return all(isinstance(item, valid_types) for item in column)

        # Filtra le colonne in base ai tipi di dati desiderati
        filtered_data = {key: value for key, value in mexp_data.items() if is_valid_column(value)}

        # Converti il dizionario filtrato in un DataFrame
        df = pd.DataFrame(filtered_data)
        df['p1_last'] = [tuple(x[-1,:]) for x in mexp_data['p1_front']]
        df['task'] = [Sign2Task[list(x.values())[0]] for x in mexp_data['task_signature']]
        df['high_target'] = [int(x[0][0]) for x in mexp_data['high_target']]
        df['dist_low'] = [x[ll][c] for x,ll,c in zip(mexp_data['layer_scores'], df['lower_ly'], df['p1_last'])]
        df['dist_up']  = [x[ll][c] for x,ll,c in zip(mexp_data['layer_scores'], df['upper_ly'], df['p1_last'])]
        df.to_csv(path.join(self.target_dir, 'data_summary.csv'), index=False)
        # Visualizza il DataFrame