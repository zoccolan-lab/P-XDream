# TODO Brief description of the experiment

import gc
import pandas as pd
from script.MaximizeActivity.plots import multiexp_lineplot, plot_optimizing_units, plot_scores, plot_scores_by_cat
from zdream.experiment import Experiment, MultiExperiment
from zdream.generator import Generator, InverseAlexGenerator
from zdream.logger import Logger, LoguruLogger
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.probe import InfoProbe, RecordingProbe
from zdream.scorer import ActivityScorer, Scorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.utils.dataset import MiniImageNet
from zdream.utils.io_ import to_gif
from zdream.utils.misc import concatenate_images, device
from zdream.utils.model import Codes, DisplayScreen, MaskGenerator, ScoringUnit, Stimuli, Score, State, mask_generator_from_template
from zdream.utils.parsing import parse_boolean_string, parse_recording, parse_scoring
from zdream.message import Message

import numpy as np
import torch
from pandas import DataFrame
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import os
from os import path
from typing import Any, Dict, List, Tuple, Type, cast


class MaximizeActivityExperiment(Experiment):

    EXPERIMENT_TITLE = "MaximizeActivity"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'

    @property
    def scorer(self)  -> ActivityScorer: return cast(ActivityScorer, self._scorer) 

    @property
    def subject(self) -> NetworkSubject:    return cast(NetworkSubject, self._subject) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'MaximizeActivityExperiment':
        '''
        Static constructor for a _MaximizeActivityExperiment class from configuration file.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: _MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: _MaximizeActivity
        '''

        # Extract specific configurations
        gen_conf = conf['generator']
        msk_conf = conf['mask_generator']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # --- MASK GENERATOR ---

        template = parse_boolean_string(boolean_str=msk_conf['template'])
        mask_generator = mask_generator_from_template(template=template, shuffle=msk_conf['shuffle'])
        
        # --- GENERATOR ---

        # Dataloader
        use_nat = template.count(False) > 0
        if use_nat:
            dataset      = MiniImageNet(root=gen_conf['mini_inet'])
            # rsz_inet_dir = resize_images(input_dir = gen_conf['mini_inet'])
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
        record_target = parse_recording(input_str=sbj_conf['rec_layers'], net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = NetworkSubject(
            record_probe=probe,
            network_name=sbj_conf['net_name']
        )
        
        sbj_net._network.eval() # TODO cannot access private attribute, make public method to call the eval

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

        log_conf['title'] = MaximizeActivityExperiment.EXPERIMENT_TITLE
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

                # Add screen fro natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(400, 400))
                    )

        # --- DATA ---
        data = {
            "dataset": dataset if use_nat else None,
            'render': conf['render'],
            'close_screen': conf.get('close_screen', False)
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
        name:           str = 'experiment'
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

        # Create a mock mask for one synthetic image 
        # to see if natural images are involved in the experiment
        mock_mask = self._mask_generator(1)
    
        self._use_natural = mock_mask is not None and sum(~mock_mask) > 0

        # Extract from Data

        self._render        = cast(bool, data['render'])
        self._close_screen  = cast(bool, data['close_screen'])
        
        if self._use_natural:
            self._dataset   = cast(MiniImageNet, data['dataset'])

    def _progress_info(self, i: int, msg : Message) -> str:

        # We add the progress information about the best
        # and the average score per each iteration
        stat_gen = msg.stats_gen

        if self._use_natural:
            stat_nat = msg.stats_nat

        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()
        if self._use_natural:
            best_nat = cast(NDArray, stat_nat['best_score']).mean()

        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        if self._use_natural:
            best_nat_str = f'{best_nat:.1f}'

        if self._use_natural:
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str} | best nat: {best_nat_str}'
        else:
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'

        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super}{desc}'

    def _init(self) -> Message:

        msg = super()._init()

        # Data structure to save best score and best image
        if self._use_natural:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
        
        # Set gif
        self._gif: List[Image.Image] = []

        # Last seen labels
        if self._use_natural:
            self._labels: List[int] = []

        
        #-- RF MAPPING --         
        #mock images for receptive field mapping (both forward and backward)
        self.nr_imgs4rf = 10
        mock_msg = Message(
            mask=np.ones(self.nr_imgs4rf, dtype=bool),
            label=[],    
        )
    
        mock_inp = torch.randn(self.nr_imgs4rf, *self.subject._inp_shape[1:], device=self.subject.device, requires_grad=True)
        #get the layers you are recording from. We will map the scoring neurons on each of them.
        #we sort the target layers by their depth in ascending order
        rec_layers =  list(self.subject._target.keys()) + ['00_input_01']
        rec_layers = sorted(rec_layers, key=lambda x: int(x.split('_')[0]))
        #ASSUMPTION: i am assuming that all scored units are from the same layer 
        
        rec_targets = self.subject.target
        #In case of conv layers, scored_units values will be a np.array of shape 3 x n, where 
        #each row corresponds to the coordinates of the unit of interest (c x h x w)
        scored_units = {k:np.expand_dims(np.array(v), axis = 0) 
                        if int(k.split('_')[0])>15 
                        else np.row_stack([rec_targets[k][i][v] for i in range(len(rec_targets[k]))])
                        for k,v in self.scorer._scoring_units.items()}

                
        mapped_layers = list(set(rec_layers) - set(scored_units.keys()))
        backward_target = {ml: scored_units for ml in mapped_layers}
        # Create the InfoProbe and attach it to the subject
        probe = InfoProbe(
            inp_shape=self.subject._inp_shape,
            rf_method='backward',
            backward_target= backward_target #pensarci sopra sul tipo di backward_target
        )

        # NOTE: For backward receptive field we need to register both
        #       the forward and backward probe hooks
        self.subject.register(probe)

        # Expose the subject to the mock input to collect the set
        # of shapes of the underlying network
        _ = self.subject((mock_inp, mock_msg), raise_no_probe=False, with_grad=True)
        
        # Collect the receptive fields from the info probe
        #NOTE: if done for many units, it takes a lot of time and memory space
        #issue with garbage collecting gradients
        #TO DO: now it seems that rec_field works on cpu(). do it on GPU
        msg.rf_maps = probe.rec_field
        
        #get rf perimeter on the input (NOTE: they are computed on network input size, that can differ
        # from  the outputted generated images. Therefore, a resizing of rfs is implemented)
        rf_on_input = msg.rf_maps[('00_input_01',msg.scr_layers[-1])]
        #rf_2p_mask will include the perimeters of the rfs
        rf_2p_mask = [np.zeros(self.generator.output_dim[-2:], dtype=bool) for _ in range(len(rf_on_input))]
        img_side_gen = self.generator.output_dim[-1]; img_side_net_input= self.subject._inp_shape[-1]
        rescale_factor = img_side_gen / img_side_net_input
        for i,rf in enumerate(rf_on_input):
            #rescale to output size from actual alexnet input (224,224) -> (256,256)
            rf_rescaled = [tuple(round(c * rescale_factor) if round(c * rescale_factor)<img_side_gen else (img_side_gen-1) 
                            for c in coord) 
                            if (idx == 1 or idx == 2) else coord
                            for idx, coord in enumerate(rf)]
            #get the borders of rfs        
            rf_2p_mask[i][rf_rescaled[1][0]:rf_rescaled[1][-1]+1, rf_rescaled[2][0]] = True
            rf_2p_mask[i][rf_rescaled[1][0]:rf_rescaled[1][-1]+1, rf_rescaled[2][-1]] = True
            rf_2p_mask[i][rf_rescaled[1][0],                      rf_rescaled[2][0]:rf_rescaled[2][-1]+1] = True
            rf_2p_mask[i][rf_rescaled[1][-1],                     rf_rescaled[2][0]:rf_rescaled[2][-1]+1] = True
        self._rf_2p_mask = rf_2p_mask
        
        # Remove the probe from the subject
        self.subject.remove(probe) 
        return msg

    def _progress(self, i: int, msg : Message):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.solution
        best_synthetic, _ = self.generator(data=(best_code, Message(mask=np.array([True]))))
        
        """#evidence the rf of conv layers only
        if 'conv' in msg.scr_layers[-1]:
            for rf_mask in self._rf_2p_mask:
                best_synthetic[:,:,rf_mask] = np.inf"""
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
        if self._use_natural:
            best_nat = self._best_nat_img
            
        
        #get rf perimeter
        if 'conv' in msg.scr_layers[-1]:
            for rf_mask in self._rf_2p_mask:
                best_gen[:,rf_mask] = np.inf
                if self._use_natural:
                    best_nat[:,rf_mask] = np.inf
        
        # Saving images
        for img, label in [
            (to_pil_image(best_gen), 'best synthetic'),
            (to_pil_image(best_nat), 'best natural'),
            (concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli'),
        ] if self._use_natural else [
            (to_pil_image(best_gen), 'best synthetic')
        ]:
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)
        
        out_fp = path.join(img_dir, 'evolving_best.gif')
        self._logger.info(f'> Saving evolving best stimuli across generations to {out_fp}')
        to_gif(image_list=self._gif, out_fp=out_fp)

        self._logger.info(mess='')
        
        # 2. Save plots

        plots_dir = path.join(self.target_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        self._logger.info(mess=f"Saving plots to {plots_dir}")
        
        self._logger.prefix='> '
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

        if self._use_natural:
            plot_scores_by_cat(
                scores=(
                    np.stack(msg.scores_gen_history),
                    np.stack(msg.scores_nat_history)
                ),
                lbls    = self._labels,
                out_dir = plots_dir, 
                dataset = self._dataset,
                logger=self._logger
            )
        self._logger.prefix=''
        
        self._logger.info(mess='')
        
        return msg
        

    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[State, Message]:

        # We save the last set of stimuli
        self._stimuli, msg = data
        if self._use_natural:
            self._labels.extend(msg.label)

        return super()._stimuli_to_sbj_state(data)

    def _stm_score_to_codes(self, data: Tuple[Score, Message]) -> Tuple[Codes, Message]:

        sub_score, msg = data

        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._stm_score_to_codes((sub_score, msg))
    
    def _sbj_state_to_scr_state(self, sbj_state : Tuple[State, Message]) -> Tuple[State, Message]:
        '''
        
        '''
        
        state, msg = sbj_state
        msg.states_history.append(state)
        
        return super()._sbj_state_to_scr_state(sbj_state=sbj_state)


# --- MULTI EXPERIMENT ---

class NeuronScalingMultiExperiment(MultiExperiment):

    def __init__(
            self, 
            experiment:      Type['Experiment'], 
            experiment_conf: Dict[str, List[Any]], 
            default_conf:    Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)

        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True
    
    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [
            DisplayScreen(title=MaximizeActivityExperiment.GEN_IMG_SCREEN, display_size=(400, 400))
        ]

        # Add screen for natural images if at least one will use it
        if any(parse_boolean_string(conf['mask_generator']['template']).count(False) > 0 for conf in self._search_config):
            screens.append(
                DisplayScreen(title=MaximizeActivityExperiment.NAT_IMG_SCREEN, display_size=(400, 400))
            )

        return screens
    
    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger
        
    def _init(self):
        super()._init()
        
        self._data['desc'   ] = 'Scores at varying number of scoring neurons' # TODO
        self._data['score'  ] = list()
        self._data['neurons'] = list()
        self._data['layer'  ] = list()
        self._data['iter'   ] = list()


    def _progress(self, exp: MaximizeActivityExperiment, msg : Message, i: int):

        super()._progress(exp, i, msg = msg)

        self._data['score']  .append(msg.stats_gen['best_score'])
        self._data['neurons'].append(exp.scorer.n_scoring_units)
        self._data['layer']  .append(list(exp.scorer._scoring_units.keys()))
        self._data['iter']   .append(exp._iteration) # TODO make public property        

    def _finish(self):
        
        super()._finish()

        plot_optimizing_units(
            multiexp_data=self._data,
            out_dir=self._logger.target_dir,
            logger=self._logger
        )


class LayersCorrelationMultiExperiment(MultiExperiment):

    def __init__(
            self, 
            experiment:      Type['Experiment'], 
            experiment_conf: Dict[str, List[Any]], 
            default_conf:    Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)
        
        #-- RANDOM UNITS SAMPLING ---

        #QUA FARE IL SAMPLING DELLE UNITà RANDOMICHE CHE POI POSSONO ESSERE USATE (SCORING) PER IL RECORDING
        
        # Create a on-the-fly network subject to extract all network layer names
        #ASSUMPION: ALL THE EXPERIMENTS SHARE THE SAME NETWORK MODEL (e.g. alexnet)
        net_info: Dict[str, Tuple[int, ...]] = NetworkSubject(network_name=self._search_config[0]['subject']['net_name']).layer_info
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
        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True

    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger
    
    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [
            DisplayScreen(title=MaximizeActivityExperiment.GEN_IMG_SCREEN, display_size=(400, 400))
        ]

        # Add screen for natural images if at least one will use it
        if any(parse_boolean_string(conf['mask_generator']['template']).count(False) > 0 for conf in self._search_config):
            screens.append(
                DisplayScreen(title=MaximizeActivityExperiment.NAT_IMG_SCREEN, display_size=(400, 400))
            )

        return screens
        
    def _init(self):

        super()._init()
        
        # Description
        self._data['desc'] = 'Layers correlations ...' # TODO

        # Delta activation
        self._data['deltaA_rec'] = list()

        # Hyperparameter and recordings
        self._data['layer']     = list()
        self._data['score']     = list()
        self._data['score_nat'] = list()
        self._data['iter']      = list()
        self._data['neurons']   = list()

        # Correlation
        self._data['Copt_rec']      = list()
        self._data['Copt_rec_var']  = list()
        self._data['Crec_rec']      = list()
        self._data['Crec_rec_var']  = list()

    
    def _progress(self, exp: MaximizeActivityExperiment, msg : Message, i: int):

        super()._progress(exp, i, msg=msg)
        mock_template = exp._mask_generator(1)
        use_nat = mock_template is not None and sum(~mock_template) > 0

        # Update score and parameters
        self._data['score']    .append(msg.stats_gen['best_score'])
        self._data['score_nat'].append(msg.stats_nat['best_score'] if use_nat else np.nan)
        self._data['neurons']  .append(exp.scorer.n_scoring_units)
        self._data['layer']    .append(msg.scr_layers)
        self._data['iter']     .append(exp._iteration) # TODO make public property
        
        # Create new dictionary for fields
        deltaA_rec  = {}
        Copt_rec    = {}
        Copt_rec_var= {}
        Crec_rec    = {}
        Crec_rec_var= {}

        # Iterate over recorded layers to get statistics about non optimized sites
        non_scoring = self._get_non_scoring_units(exp=exp)
        
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
                corr_vec = np.corrcoef(msg.stats_gen['mean_shist'], avg_rec)[0, 1:]
                Copt_rec[layer]     = np.mean(corr_vec)
                Copt_rec_var[layer] = np.std(corr_vec)

                # Compute the correlation between each avg recorded units. 
                # Store correlation mean and variance
                Crec_rec_mat = np.corrcoef(avg_rec)
                up_tria_idxs = np.triu_indices(Crec_rec_mat.shape[0], k=1)
                Crec_rec[layer]     = np.mean(Crec_rec_mat[up_tria_idxs])
                Crec_rec_var[layer] = np.std(Crec_rec_mat[up_tria_idxs])

        # Update correlations
        self._data['deltaA_rec']  .append(deltaA_rec)
        self._data['Copt_rec']    .append(Copt_rec)
        self._data['Copt_rec_var'].append(Copt_rec_var)
        self._data['Crec_rec']    .append(Crec_rec)
        self._data['Crec_rec_var'].append(Crec_rec_var)

    def _finish(self):

        super()._finish()

        # Organize results as a dataframe
        df = self._create_df(multiexp_data=self._data)

        # Create output image folder
        plots_dir = path.join(self.target_dir, 'plots')
        self._logger.info(mess=f'Creating directory {plots_dir}')
        os.makedirs(plots_dir)

        self._logger.prefix = '> '

        # Plot neuron score scaling
        sc_metrics = ['scores','scores_norm']
        for m in sc_metrics:
            multiexp_lineplot(
                out_df=df, 
                gr_vars= ['layers', 'neurons'],
                out_dir=plots_dir,
                y_var = m,
                metrics = ['mean', 'sem'],
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
        
        self._logger.prefix = ''
        
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
        rec_keys = ['deltaA_rec', 'Copt_rec', 'Copt_rec_var', 'Crec_rec', 'Crec_rec_var']

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

        #for layer, scoring in exp.scorer.target.items():
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





