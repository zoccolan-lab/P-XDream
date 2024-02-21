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

from tqdm import trange
from torch import Tensor
from typing import cast, Tuple
from numpy.typing import NDArray
from zdream.experiment import Experiment, ExperimentConfig

from zdream.utils import Logger, Stimuli, preprocess_image
from zdream.utils import SubjectState

from zdream.utils import Message, read_json
from zdream.scores import MSEScorer
from zdream.optimizer import GeneticOptimizer
from zdream.generator import InverseAlexGenerator
from zdream.subject import NetworkSubjectAbstract

from zdream.utils import device

class TrivialSubject(NetworkSubjectAbstract):
    
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
    
    def __call__(
        self,
        data : Tuple[Stimuli, Message]
    ) -> Tuple[SubjectState, Message]:
        
        img, msg = data
    
        return {self._name : img.cpu().numpy()}, msg
    
""" TODO include in the experiment
# Compute the MSE loss to show feedback to used
mse = min([mean_squared_error(img, target_image[0]) for img in sub_state['image']])

# Get the current best score to update the progress bar
stat = optim.stats
best = cast(NDArray, stat['best_score']).mean()
curr = cast(NDArray, stat['curr_score']).mean()
desc = f'Generation {gen} | best score: {best:.1f} | avg score: {curr:.1f} | MSE: {mse:.3f}'
progress.set_description(desc) 
"""

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
    
    subject=TrivialSubject(name='image')
    
    experiment_config = ExperimentConfig(
        generator=generator,
        scorer=scorer,
        optimizer=optim,
        subject=subject,
        logger=Logger(),
        num_gen=num_gens
    )
    
    experiment = Experiment(config=experiment_config)
    
    experiment.run()

    # Save the best performing image to file
    best_state = experiment.optimizer.solution
    best_image, msg = experiment.generator(best_state)

    save_image = make_grid([*torch.from_numpy(target_image), *best_image.cpu()], nrow=2)
    save_image = cast(Image.Image, to_pil_image(save_image))
    
    save_dir_fp = path.join(args.save_dir, 'target_recovery')
    os.makedirs(save_dir_fp, exist_ok=True)
    
    img_name = path.splitext(path.basename(args.test_img))[0]
    save_img_fp = path.join(save_dir_fp, f'{img_name}_{args.gen_variant}.png')

    save_image.save(save_img_fp)
    
    Image.open(save_img_fp).show()
    
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