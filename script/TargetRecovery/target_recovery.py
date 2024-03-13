from os import path
import os
import torch
import numpy as np

from torch import Tensor
from typing import Any, Dict, List, cast, Tuple
from numpy.typing import NDArray
from zdream.logger import Logger, LoguruLogger
from zdream.experiment import Experiment
from zdream.utils.io_ import to_gif
from zdream.utils.misc import concatenate_images, preprocess_image
from zdream.utils.model import DisplayScreen, MaskGenerator, Message, RecordingUnit, Stimuli, StimuliScore

from zdream.utils.model import SubjectState
from zdream.utils.misc import device

from zdream.scores import MSEScorer, Scorer
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.generator import Generator, InverseAlexGenerator
from zdream.subject import InSilicoSubject

from PIL import Image
from torchvision.transforms.functional import to_pil_image

class _TrivialSubject(InSilicoSubject):
    
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        
    def __str__(self) -> str:
        return f"TrivialSubject[layer_name: {self._name}]"
    
    def __call__(
        self,
        data : Tuple[Stimuli, Message]
    ) -> Tuple[SubjectState, Message]:
        
        img, msg = data

        state = {self._name : img.cpu().numpy()}

        self._states.append(state)
    
        return state, msg
    
    @property
    def target(self) -> Dict[str, RecordingUnit]:
        return {self._name: None}
    

    
    
class _TargetRecoveryExperiment(Experiment):

    EXPERIMENT_TITLE = 'TargetRecovery'

    TARGET_SCREEN = 'Evolving Target'

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> '_TargetRecoveryExperiment':
        
        # Extract specific configurations
        gen_conf = conf['generator']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        target_image = preprocess_image(
            image_fp=scr_conf['img_path'], 
            resize=(256, 256)
        )

        scorer = MSEScorer(
            target={'image': target_image}
        )

        generator = InverseAlexGenerator(
            root = gen_conf['weights'],
            variant = gen_conf['variant'],
            output_pipe=transform,
        ).to(device)

        optim = GeneticOptimizer(
            states_shape   = generator.input_dim,
            random_seed    =     conf['random_seed'],
            random_distr   = opt_conf['random_distr'],
            mutation_rate  = opt_conf['mutation_rate'],
            mutation_size  = opt_conf['mutation_size'],
            population_size= opt_conf['pop_size'],
            temperature    = opt_conf['temperature'],
            num_parents    = opt_conf['num_parents']
        )
        
        subject = _TrivialSubject(name='image')

        log_conf['title'] = _TargetRecoveryExperiment.EXPERIMENT_TITLE
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
                    screen=DisplayScreen(title=cls.TARGET_SCREEN, display_size=(400, 400))
                )

        # --- DATA ---
        data = {
            'render': conf['render'],
            'close_screen': conf.get('close_screen', False)
        }
        
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = subject,
            logger         = logger,
            iteration      = conf['iter'],
            name           = log_conf['name'],
            data           = data
        )

        return experiment

    def __init__(
        self, 
        generator: Generator, 
        subject: InSilicoSubject, 
        scorer: Scorer, optimizer: 
        Optimizer, iteration: int, 
        logger: Logger,
        mask_generator: MaskGenerator | None = None, 
        data: Dict[str, Any] = dict(),
        name: str = 'target_recovery_experiment'
    ) -> None:
        
        super().__init__(
            generator=generator, 
            subject=subject, 
            scorer=scorer, 
            optimizer=optimizer, 
            iteration=iteration, 
            logger=logger, 
            mask_generator=mask_generator, 
            data=data, 
            name=name
        )

        self._render        = cast(bool, data['render'])
        self._close_screen  = cast(bool, data['close_screen'])
        
    @property
    def scorer(self) -> MSEScorer: return cast(MSEScorer, self._scorer)
    
    def _sbj_state_to_stm_score(self, data: Tuple[SubjectState, Message]) -> Tuple[StimuliScore, Message]:
        
        self._state, _ = data
        return self.scorer(data=data)
            
    def _progress_info(self, i: int) -> str:

        trg_img = self.scorer.template['image']

        mse = min([
            float(MSEScorer.mse(
                trg_img, 
                np.expand_dims(img, axis=0) 
            ))
            for img in self._state['image']
        ])
        
        stat = self.optimizer.stats
        best = cast(NDArray, stat['best_score']).mean()
        curr = cast(NDArray, stat['curr_score']).mean()
        
        desc = f' | Best score: {best:>7.3f} | Avg score: {curr:>7.3f} | MSE: {mse:.5f}'
        
        progress_super = super()._progress_info(i=i)
        return f'{progress_super}{desc}'
    
    def _init(self):

        super()._init()
        
        # Set gif
        self._gif: List[Image.Image] = []

    
    def _progress(self, i: int):

        super()._progress(i)

        # Get best stimuli
        best_code = self.optimizer.solution
        best_synthetic, _ = self.generator(codes=best_code, pipeline=False)
        best_synthetic_img = to_pil_image(best_synthetic[0])
            
        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )

        if self._render:

            self._logger.update_screen(
                screen_name=self.TARGET_SCREEN,
                image=best_synthetic_img
            )
    
    def _finish(self):
        
        super()._finish()
    
        # Save the best performing image to file
        best_state = self.optimizer.solution
        best_image, _ = self.generator(codes=best_state, pipeline=False)

        trg_img = torch.tensor(self.scorer.template['image'])
        concatenated = concatenate_images(img_list=[best_image[0], trg_img[0]])
        
        img_dir = path.join(self.target_dir, 'images')
        self._logger.info(mess=f'Creating directory {img_dir}')
        os.makedirs(img_dir, exist_ok=True)

        img_fp = path.join(img_dir, 'best_stimuli.png')
        self._logger.info(mess=f"> Saving best image to {img_fp}")
        concatenated.save(img_fp)

        gif_fp = path.join(img_dir, 'evolving_best.gif')
        self._logger.info(f'> Saving evolving best stimuli across generations to {gif_fp}')
        to_gif(image_list=self._gif, out_fp=gif_fp)


def transform(
    imgs : Tensor,
    mean : Tuple[int, ...] = (104.0, 117.0, 123.0), # type: ignore
    raw_scale : float = 255.
) -> Tensor:
    mean : Tensor = torch.tensor(mean, device=imgs.device).reshape(-1, 1, 1)

    imgs += mean
    imgs /= raw_scale

    return imgs.clamp(0, 1)
