"""
TODO Experiment description
"""

import glob
import os
from os import path
import numpy as np
from tqdm import trange
from argparse import ArgumentParser
import torch
from torch import Tensor
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from typing import cast, Tuple
from collections import defaultdict
from loguru import logger
from functools import partial
import random
import matplotlib

from zdream.generator import InverseAlexGenerator
from zdream.model import Codes, Logger, Message, Stimuli, StimuliScore, SubjectState
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject
from zdream.utils import *
from zdream.plot_utils import *
from zdream.scores import MaxActivityScorer
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.experiment import Experiment, ExperimentConfig

matplotlib.use('TKAgg')


class _MaximizeActivity(Experiment):

    def __init__(self, config: ExperimentConfig, name: str = "maximize_activity") -> None:
        
        super().__init__(config=config, name=name)
        self._data = cast(Dict[str, Any], config.data)
        
    def _progress_info(self, i: int) -> str:
        
        # We add the progress information about the best
        # and the average score per each iteration
        stat = self.optimizer.stats
        
        best = cast(NDArray, stat['best_score']).mean()
        curr = cast(NDArray, stat['curr_score']).mean()
        
        desc = f' | best score: {best:.1f} | avg score: {curr:.1f}'
        
        progress_super = super()._progress_info(i=i)
        
        return f'{progress_super}{desc}'
        
    def _init(self):
        
        super()._init()
        
        # Data structure to save best score and best image
        self._best_scr = {'gen': 0, 'nat': 0}
        self._best_img = {
            k: torch.zeros(self.generator.output_dim)
            for k in ['gen', 'nat']
        }
        
        # Set screen
        self._screen_syn = "Best synthetic image"
        self._logger.add_screen(screen_name=self._screen_syn, display_size=(800, 400))
        
        # Set screen
        # self._screen_nat = "Best natural image"
        # self._logger.add_screen(screen_name=self._screen_nat)
        
    def _progress(self, i: int):
        
        super()._progress(i)
        
        # Get best stimuli
        best_code = self.optimizer.solution
        best_synthetic, _ = self.generator(best_code)
        best_natural = self._best_img['nat']
        
        out_image = concatenate_images(img_list=[best_synthetic[0], best_natural])
        
        self._logger.update_screen(
            screen_name=self._screen_syn,
            image=out_image
        )
        
    def _finish(self):
        
        super()._finish()
        
        # Close screens
        self._logger.remove_all_screens()
        
        # 1) Best Images
        
        # We create the directory to store results
        out_dir_fp = path.join(self._data['save_dir'], self._name)
        os.makedirs(out_dir_fp, exist_ok=True)
        
        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_code = self.optimizer.solution
        best_synthetic, _ = self.generator(best_code)
        
        # We retrieve the stored best natural image
        best_natural = self._best_img['nat']
        # best_synthetic = self._best_img['gen']
        
        # We concatenate them
        out_image = concatenate_images(img_list=[best_synthetic[0], best_natural])
        
        # We store them
        out_image_fp = path.join(out_dir_fp, f'best_images.png')
        self._logger.info(mess=f"Saving best images to {out_image_fp}")
        out_image.save(out_image_fp)    
        
        # 2) Score plots
        plot_optimization_profile(self._optimizer, save_dir = out_dir_fp)
        plot_scores_by_cat(self._optimizer, self._generator._lbls_nat_presented, save_dir = out_dir_fp)
        
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        
        # We save the last set of stimuli
        self._stimuli, _ = data
        
        return super()._stimuli_to_sbj_state(data)
    
    def _stm_score_to_codes(self, data: Tuple[StimuliScore, Message]) -> Codes:
        
        sub_score, msg =data
        
        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        for imtype, mask in zip(['gen', 'nat'], [msg.mask, ~msg.mask]):
            
            max_, argmax = tuple(f_func(sub_score[mask]) for f_func in [np.amax, np.argmax])
        
            if max_ > self._best_scr[imtype]:
                self._best_scr[imtype] = max_
                self._best_img[imtype] = self._stimuli[torch.tensor(mask)][argmax]
        
        return super()._stm_score_to_codes((sub_score, msg))

    

def transform(
    imgs : Tensor,
    mean : Tuple[int, ...] = (104.0, 117.0, 123.0), # type: ignore
    raw_scale : float = 255.
) -> Tensor:
    """ Generator output pipe transformation """
    
    mean : Tensor = torch.tensor(mean, device=imgs.device).reshape(-1, 1, 1)

    imgs += mean
    imgs /= raw_scale

    return imgs.clamp(0, 1)

def main(args):
    
    # Transform Mask sequence into boolean
    base_seq = [char == 't' for char in args.mask_template] # TODO It doesn't check for others letter instead of 'f'
    
    # Create network subject
    sbj_net = NetworkSubject(network_name='alexnet')
    
    layer_names = sbj_net.layer_names
    
    # Extract the name of recording layers
    rec_layers = [layer_names[i] for i in args.rec_layers] # TODO Check input as a tuple
    
    # Extract scoring layers from arguments
    score_dict = eval(args.score_layers) # TODO Study other way, what if args.score_layers is 'os.remove("C:")' ?
    
    # Building target neurons
    # TODO what is this ifelse doing?
    score_dict = {
        layer_names[k]: (v[1] if isinstance(v[1], list)  else random.sample(range(1000), v[0]))
        for k, v in score_dict.items()
    }
    print(score_dict)
    # Generator with Dataloader
    mini_IN = MiniImageNet(root=args.tiny_inet_root)
    mini_IN_loader = DataLoader(mini_IN, batch_size=2, shuffle=True)
    # mini_IN_loader = DataLoader(RandomImageDataset(1000, (3, 256, 256)), batch_size=2, shuffle=True)
    
    generator = InverseAlexGenerator(
        root=args.gen_root,
        variant=args.gen_variant,
        #output_pipe=transform,
        nat_img_loader=mini_IN_loader
    ).to(device)
    
    # Initialize  NetworkSubject with a recording probe
    
    record_target = {l: None for l in rec_layers} # Record from any layers
    probe = RecordingProbe(target = record_target) # type: ignore TODO check typing
    
    sbj_net = NetworkSubject(
        record_probe=probe, 
        network_name='alexnet'
    )
    sbj_net._network.eval() # TODO cannot access private attribute, make public method to call the eval
    
    # Scorer
    aggregate = lambda x: np.mean(np.stack(list(x.values())), axis=0)
    scorer = MaxActivityScorer(
        trg_neurons=score_dict,
        aggregate=aggregate
    )

    # Optimizer
    optim = GeneticOptimizer(
        states_shape=generator.input_dim,
        random_state=args.optimizer_seed,
        random_distr='normal',
        mutation_rate=args.mutation_rate,
        mutation_size=args.mutation_size,
        population_size=args.pop_sz,
        temperature=args.temperature,
        num_parents=args.num_parents
    )
    
    # Mask generator
    mask_generator = partial(repeat_pattern, base_seq=base_seq, shuffle=args.mask_is_random)
    
    # Additional data
    data = {
    "save_dir": args.save_dir
    }
    
    # Experiment configuration
    experiment_config = ExperimentConfig(
        generator=generator,
        scorer=scorer,
        optimizer=optim,
        subject=sbj_net,
        logger=LoguruLogger(),
        iteration=args.num_gens,
        mask_generator=mask_generator,
        data=data
    )
    
    # Experiment
    experiment = _MaximizeActivity(experiment_config)
    experiment.run()


if __name__ == '__main__':
    
    # Loading `local_settings.json` for custom local settings
    local_folder = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, 'local_settings.json')
    script_settings = read_json(path=script_settings_fp)
    
    gen_root   = script_settings['inverse_alex_net']
    test_image = script_settings['test_image']
    image_out  = script_settings['image_out']
    tiny_inet_root = script_settings['tiny_inet_root']
    
    parser = ArgumentParser()
    
    parser.add_argument('-pop_sz',         type=int,   default=20,               help='Number of images per generation')
    parser.add_argument('-num_gens',       type=int,   default=10,               help='Number of total generations to evolve')
    parser.add_argument('-img_size',       type=tuple, default=(256, 256),       help='Size of a given image', nargs=2)
    parser.add_argument('-gen_variant',    type=str,   default='fc8',            help='Variant of InverseAlexGenerator to use')
    parser.add_argument('-optimizer_seed', type=int,   default=31415,            help='Random seed in GeneticOptimizer')
    parser.add_argument('-mutation_rate',  type=float, default=0.3,              help='Mutation rate in GeneticOptimizer')
    parser.add_argument('-mutation_size',  type=float, default=0.3,              help='Mutation size in GeneticOptimizer')
    parser.add_argument('-num_parents',    type=int,   default=2,                help='Number of parents in GeneticOptimizer')
    parser.add_argument('-temperature',    type=float, default=1.0,              help='Temperature in GeneticOptimizer')
    parser.add_argument('-mask_template',  type=str ,  default='tffff',          help='String of True(t) and False(f). It will be converted in the basic sequence of the mask')
    parser.add_argument('-mask_is_random', type=bool , default=False,            help='Defines if the mask is pseudorandom or not')
    parser.add_argument('-rec_layers',     type=tuple, default=(18,20),          help='Layers you want to record from (each int in tuple = nr of the layer)')
    parser.add_argument('-score_layers',   type=str,   default="{20:(1,None)}",  help='Layers you want to score from (dictionary with layer nr: (nr of units, units ID (r = rand)))')

    parser.add_argument('-gen_root',       type=str,   default=gen_root,         help='Path to root folder of generator checkpoints')
    parser.add_argument('-tiny_inet_root', type=str,   default=tiny_inet_root,   help='Path to tiny imagenet dataset')
    parser.add_argument('-save_dir',       type=str,   default=image_out,        help='Path to store best solution')
    
    args = parser.parse_args()
    
    main(args)
    
