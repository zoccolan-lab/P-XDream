from os import path
import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from argparse import ArgumentParser
from skimage.metrics import mean_squared_error
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from loguru import logger

from tqdm import trange
from torch import Tensor
from typing import Any, Dict, cast, Tuple
from numpy.typing import NDArray
from zdream.experiment import Experiment, ExperimentConfig

from zdream.utils import Logger, Stimuli, SubjectScore, preprocess_image
from zdream.utils import SubjectState

from zdream.utils import Message, read_json
from zdream.scores import MSEScorer
from zdream.optimizer import GeneticOptimizer
from zdream.generator import InverseAlexGenerator
from zdream.subject import InSilicoSubject

from zdream.utils import device

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
    
        return {self._name : img.cpu().numpy()}, msg
    
class _LoguruLogger(Logger):
    
    def info(self, mess: str): logger.info(mess)
    def warn(self, mess: str): logger.warning(mess)
    def err (self, mess: str): logger.error(mess)
    
class _TargetRecoveryExperiment(Experiment):
    
    def __init__(self, config: ExperimentConfig, name: str = "") -> None:
        super().__init__(config=config, name=name)
        self._data = cast(Dict[str, Any], config.data)
        
    @property
    def scorer(self) -> MSEScorer: return cast(MSEScorer, self._scorer)
    
    def sbj_state_to_sbj_score(self, data: Tuple[SubjectState, Message]) -> Tuple[SubjectScore, Message]:
        
        self._state, _ = data
        return self.scorer(data=data)
            
    def progress_info(self, gen: int) -> str:
        
        trg_img = self.scorer.target['image']
        
        mse = min([
            mean_squared_error(
                trg_img, 
                np.expand_dims(img, axis=0)  # add batch size
            )
            for img in self._state['image']]
        )
        
        
        stat = self.optimizer.stats
        best = cast(NDArray, stat['best_score']).mean()
        curr = cast(NDArray, stat['curr_score']).mean()
        
        desc = f' | Best score: {best:>7.3f} | Avg score: {curr:>7.3f} | MSE: {mse:.5f}'
        
        progress_super = super().progress_info(gen=gen)
        return f'{progress_super}{desc}'
    
    def _finish(self):
        
        super()._finish()
    
        # Save the best performing image to file
        best_state = self.optimizer.solution
        best_image, msg = self.generator(best_state)

        trg_img = self.scorer.target['image']
        
        save_image = make_grid([*torch.from_numpy(trg_img), *best_image.cpu()], nrow=2)
        save_image = cast(Image.Image, to_pil_image(save_image))
        
        save_dir_fp = path.join(self._data['save_dir'], self._name)
        os.makedirs(save_dir_fp, exist_ok=True)
        
        save_img_fp = path.join(save_dir_fp, f'{self._data["out_name"]}.png')
        
        self._logger.info(mess=f"Saving best image to {save_img_fp}")

        save_image.save(save_img_fp)



def transform(
    imgs : Tensor,
    mean : Tuple[int, ...] = (104.0, 117.0, 123.0), # type: ignore
    raw_scale : float = 255.
) -> Tensor:
    mean : Tensor = torch.tensor(mean, device=imgs.device).reshape(-1, 1, 1)

    imgs += mean
    imgs /= raw_scale

    return imgs.clamp(0, 1)

def main(args):
    num_imgs = args.num_imgs
    num_gens = args.num_gens
    img_size = args.img_size
    gen_root = args.gen_root
    
    target_image = preprocess_image(image_fp=args.test_img, resize=img_size)

    scorer = MSEScorer(
        target={'image' : target_image}
    )

    generator = InverseAlexGenerator(
        root=gen_root,
        variant=args.gen_variant,
        output_pipe=transform,
    ).to(device)

    optim = GeneticOptimizer(
        states_shape=generator.input_dim,
        random_state=args.optimizer_seed,
        random_distr='normal',
        mutation_rate=args.mutation_rate,
        mutation_size=args.mutation_size,
        population_size=num_imgs,
        temperature=args.temperature,
        num_parents=args.num_parents,
    )
    
    subject=_TrivialSubject(name='image')
    
    data = {
        "save_dir": args.save_dir,
        "out_name": f'{path.splitext(path.basename(args.test_img))[0]}_{args.gen_variant}'
    }
    
    experiment_config = ExperimentConfig(
        generator=generator,
        scorer=scorer,
        optimizer=optim,
        subject=subject,
        logger=_LoguruLogger(),
        num_gen=num_gens,
        data=data
    )
    
    experiment = _TargetRecoveryExperiment(config=experiment_config, name="target-recovery")
    
    experiment.run()
    
if __name__ == '__main__':
    
    # Loading `local_settings.json` for custom local settings
    local_folder = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, 'local_settings.json')
    script_settings = read_json(path=script_settings_fp)
    
    gen_root   = script_settings['inverse_alex_net']
    test_image = script_settings['test_image']
    image_out  = script_settings['image_out']
    
    parser = ArgumentParser()
    
    parser.add_argument('-num_imgs',       type=int,   default=20,           help='Number of images per generation')
    parser.add_argument('-num_gens',       type=int,   default=250,          help='Number of total generations to evolve')
    parser.add_argument('-img_size',       type=tuple, default=(256, 256),   help='Size of a given image', nargs=2)
    parser.add_argument('-gen_variant',    type=str,   default='fc8',        help='Variant of InverseAlexGenerator to use')
    parser.add_argument('-optimizer_seed', type=int,   default=31415,        help='Random seed in GeneticOptimizer')
    parser.add_argument('-mutation_rate',  type=float, default=0.3,          help='Mutation rate in GeneticOptimizer')
    parser.add_argument('-mutation_size',  type=float, default=0.3,          help='Mutation size in GeneticOptimizer')
    parser.add_argument('-num_parents',    type=int,   default=2,            help='Number of parents in GeneticOptimizer')
    parser.add_argument('-temperature',    type=float, default=1.0,          help='Temperature in GeneticOptimizer')
    
    parser.add_argument('-gen_root',       type=str,   default=gen_root,     help='Path to root folder of generator checkpoints')
    parser.add_argument('-test_img',       type=str,   default=test_image,   help='Path to test image')
    parser.add_argument('-save_dir',       type=str,   default=image_out,    help='Path to store best solution')
    
    args = parser.parse_args()
    
    main(args)