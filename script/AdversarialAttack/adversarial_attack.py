

from os import path
import os
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple, cast
from torchvision.transforms.functional import to_pil_image

from zdream.experiment import Experiment
from zdream.generator import Generator, InverseAlexGenerator
from zdream.logger import Logger, LoguruLogger
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.scorer import Scorer, WeightedPairSimilarityScorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.io_ import to_gif
from zdream.utils.misc import concatenate_images, device
from zdream.utils.dataset import MiniImageNet
from zdream.utils.parsing import parse_boolean_string, parse_recording, parse_scoring, parse_signature
from zdream.utils.model import Codes, DisplayScreen, MaskGenerator, ScoringUnit, Stimuli, StimuliScore, SubjectState, mask_generator_from_template
from zdream.message import Message

from numpy.typing import NDArray

class AdversarialAttackExperiment(Experiment):

    EXPERIMENT_TITLE = "MaximizeActivity"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'

    @property
    def scorer(self)  -> WeightedPairSimilarityScorer: return cast(WeightedPairSimilarityScorer, self._scorer) 

    @property
    def subject(self) -> NetworkSubject:    return cast(NetworkSubject, self._subject) 
    
    @property
    def num_imgs(self) -> int: return self._n_group * self._optimizer.n_states

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'AdversarialAttackExperiment':
        '''
        Static constructor for a _AdversarialAttackExperiment class from configuration file.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: _AdversarialAttackExperiment instance with hyperparameters set from configuration.
        :rtype: _AdversarialAttackExperiment
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
            dataset    = MiniImageNet(root=gen_conf['mini_inet'])
            dataloader = DataLoader(dataset, batch_size=gen_conf['batch_size'], shuffle=True)

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

        # TODO: Implement parsing for signature
        signature = parse_signature(
            input_str=scr_conf['signature'],
            net_info=layer_info,
        )

        scorer = WeightedPairSimilarityScorer(
            signature=signature,
            trg_neurons=scoring_units,
            metric=scr_conf['metric'],
            dist_reduce_fn=None,
            layer_reduce_fn=None,
        )

        # --- OPTIMIZER ---

        optim = GeneticOptimizer(
            states_shape   = (2, *generator.input_dim),
            random_seed    =     conf['random_seed'],
            random_distr   = opt_conf['random_distr'],
            mutation_rate  = opt_conf['mutation_rate'],
            mutation_size  = opt_conf['mutation_size'],
            population_size= opt_conf['pop_size'],
            temperature    = opt_conf['temperature'],
            num_parents    = opt_conf['num_parents']
        )

        #  --- LOGGER --- 

        log_conf['title'] = AdversarialAttackExperiment.EXPERIMENT_TITLE
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
            'display_plots': conf['display_plots'],
            'render': conf['render'],
            'close_screen': conf.get('close_screen', False),
            'n_group' : conf['n_group'],
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
    
        self._use_natural = mock_mask is not None and mock_mask.size and sum(~mock_mask) > 0

        # Extract from Data

        self._display_plots = cast(bool, data['display_plots'])
        self._render        = cast(bool, data['render'])
        self._close_screen  = cast(bool, data['close_screen'])
        self._n_group       = cast(int,  data['n_group'])
        
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

        return msg

    def _progress(self, i: int, msg : Message):

        super()._progress(i, msg)

        # Get best stimuli
        best_code = msg.solution
        best_synthetic, _ = self.generator(
            data=(best_code, Message(mask=np.array([True]))),
        )
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
        best_gen, _ = self.generator(
            data=(msg.solution, Message(mask=np.array([True]))),
        )
        best_gen = best_gen[0] # remove 1 batch size

        # We retrieve the stored best natural image
        if self._use_natural:
            best_nat = self._best_nat_img

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

        if self._use_natural:
            pass

        self._logger.prefix=''
        
        self._logger.info(mess='')
        
    def _run_init(self, msg : Message) -> Tuple[Codes, Message]:
        codes, msg = super()._run_init(msg)
        
        msg.n_group = self._n_group
        
        return codes, msg
        
    def _codes_to_inputs(self, data: Tuple[Codes, Message]) -> Tuple[Codes, Message]:
        # In the adversarial attack experiment each code represent
        # a pair of images, so is expected to have double the size
        # required by the generator, we override this hook to properly
        # resize the codes to split them in half and stack them along
        # the batch dimension
        
        codes, msg = data
        
        codes = rearrange(codes, f'b g ... -> (b g) ...', g=self._n_group)
    
        return codes, msg
    
    def _sbj_state_to_scr_state(self, sbj_state: Tuple[SubjectState, Message]) -> Tuple[SubjectState, Message]:
        
        state, msg = sbj_state
        state = {k : rearrange(v, '(b g) ... -> b g ...', g=self._n_group) for k, v in state.items()}
        
        return state, msg

    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:

        # We save the last set of stimuli
        self._stimuli, msg = data
        if self._use_natural:
            self._labels.extend(msg.label)

        return super()._stimuli_to_sbj_state(data)

    def _stm_score_to_codes(self, data: Tuple[StimuliScore, Message]) -> Tuple[Codes, Message]:

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

