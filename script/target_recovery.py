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

from zdream.utils import Stimuli
from zdream.utils import SubjectState
from zdream.utils import SubjectScore

from zdream.utils import Message
from zdream.scores import MSEScore
from zdream.optimizer import GeneticOptimizer
from zdream.generator import InverseAlexGenerator

from zdream.utils import device

def trivial_subj(
    data : Tuple[Stimuli, Message],
    name : str = 'image',
) -> Tuple[SubjectState, Message]:
    img, msg = data
    
    return {name : img.cpu().numpy()}, msg

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

    target_image = Image.open('/home/pmurator/Downloads/test_image.jpg')
    target_image = np.asarray(target_image.resize(img_size)) / 255.
    target_image = rearrange(target_image, 'h w c -> 1 c h w')

    score = MSEScore(
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

    # Initialize optimizer with random condition
    # and produce initial (stimuli, msg)
    opt_state = optim.init()

    progress = trange(num_gens, desc='Generation 0 | best score: --- | avg score: --- | MSE: ---')
    for gen in progress:
        # Use current optimizer states to produce new images
        stimuli, msg = generator(opt_state)
        
        # Convert stimuli to subject state using the trivial subject
        sub_state, msg = trivial_subj(
            data=(stimuli, msg),
            name='image'
        )

        # Use scorer to score the newly computed subject states
        sub_score, msg = score(
            data=(sub_state, msg)
        )
        
        # Use the score to step the optimizer
        opt_state = optim.step(
            data=(sub_score, msg)
        )

        # Compute the MSE loss to show feedback to used
        mse = min([mean_squared_error(img, target_image[0]) for img in sub_state['image']])
        
        # Get the current best score to update the progress bar
        stat = optim.stats
        best = cast(NDArray, stat['best_score']).mean()
        curr = cast(NDArray, stat['curr_score']).mean()
        desc = f'Generation {gen} | best score: {best:.1f} | avg score: {curr:.1f} | MSE: {mse:.3f}'
        progress.set_description(desc)
        
    # Save the best performing image to file
    best_state = optim.solution
    best_image, msg = generator(best_state)

    save_image = make_grid([*torch.from_numpy(target_image), *best_image.cpu()], nrow=2)
    save_image = cast(Image.Image, to_pil_image(save_image))

    save_image.save(args.save_name)
    
if __name__ == '__main__':
    gen_root = '/media/pmurator/archive/InverseAlexGenerator'
    
    parser = ArgumentParser()
    
    parser.add_argument('-num_imgs', type=int, default=20,  help='Number of images per generation')
    parser.add_argument('-num_gens', type=int, default=250, help='Number of total generations to evolve')
    parser.add_argument('-img_size', type=tuple, nargs=2, default=(256, 256), help='Size of a given image')
    
    parser.add_argument('-gen_root', type=str, default=gen_root, help='Path to root folder of generator checkpoints')
    parser.add_argument('-gen_variant', type=str, default='fc7', help='Variant of InverseAlexGenerator to use')
    
    parser.add_argument('-optimizer_seed', type=int, default=31415, help='Random seed in GeneticOptimizer')
    parser.add_argument('-mutation_rate', type=float, default=0.3, help='Mutation rate in GeneticOptimizer')
    parser.add_argument('-mutation_size', type=float, default=0.3, help='Mutation size in GeneticOptimizer')
    parser.add_argument('-temperature', type=float, default=1.0, help='Temperature in GeneticOptimizer')
    parser.add_argument('-num_parents', type=int, default=2, help='Number of parents in GeneticOptimizer')
    
    parser.add_argument('-save_name', type=str, default='result/output/target_recovery.jpg', help='Path to store best solution')
    
    args = parser.parse_args()
    
    main(args)