from abc import ABC
from os import path
from matplotlib import pyplot as plt
import numpy as np
import glob
from tqdm import trange
from argparse import ArgumentParser
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from typing import cast, Tuple
from collections import defaultdict
from loguru import logger
from functools import partial

from zdream.generator import InverseAlexGenerator
from zdream.model import Codes, Logger, Message, Stimuli, StimuliScore, SubjectState
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject
from zdream.utils import *
from zdream.scores import MaxActivityScorer
from zdream.optimizer import GeneticOptimizer
from zdream.experiment import Experiment, ExperimentConfig

class _MiniImageNet(ImageFolder):

    def __init__(self, root, transform=transforms.Compose([transforms.Resize((256, 256)),  
    transforms.ToTensor()]), target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        #load the .txt file containing imagenet labels (all 1000 categories)
        lbls_txt = glob.glob(os.path.join(root, '*.txt'))
        with open(lbls_txt[0], "r") as f:
            lines = f.readlines()
        self.label_dict = {line.split()[0]: 
                        line.split()[2].replace('_', ' ')for line in lines}
    #maintain this method here?
    def class_to_lbl(self,lbls : Tensor): #takes in input the labels and outputs their categories
        return [self.label_dict[self.classes[lbl]] for lbl in lbls.tolist()]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)[0]

class _LoguruLogger(Logger):
    
    def info(self, mess: str): logger.info(mess)
    def warn(self, mess: str): logger.warning(mess)
    def err (self, mess: str): logger.error(mess)
    
class _MaximizeActivity(Experiment):
    def __init__(self, config: ExperimentConfig, name: str = "maximize_activity") -> None:
        super().__init__(config=config, name=name)
        self._data = cast(Dict[str, Any], config.data)
        
    def progress_info(self, i: int) -> str:
        
        stat = self.optimizer.stats
        best = cast(NDArray, stat['best_score']).mean()
        curr = cast(NDArray, stat['curr_score']).mean()
        desc = f' | best score: {best:.1f} | avg score: {curr:.1f}'
        progress_super = super()._progress_info(i=i)
        
        return f'{progress_super}{desc}'
        
    def _init(self):
        
        super()._init()
        
        self._rec_dict = defaultdict(list)
        self._best = {'gen': 0, 'nat': 0}
        self._best_img = {'gen': torch.tensor([]), 'nat': torch.tensor([])}
        
    def _finish(self):
        
        super()._finish()
        
        # Save the best performing image to file
        best_state = self.optimizer.solution
        best_image, msg = self.generator(best_state)
                
        save_dir_fp = path.join(self._data['save_dir'], self._name)
        os.makedirs(save_dir_fp, exist_ok=True)
        
        fix, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(self.optimizer.stats['best_shist'], label='Synthetic')
        ax[0].plot(self.optimizer.stats_nat['best_shist'], label='Natural')
        ax[0].set_xlabel('Generation cycles')
        ax[0].set_ylabel('Max Target Activations')
        ax[0].set_title('Better than Ponce...')
        ax[0].legend()

        ax[1].plot(self.optimizer.stats['mean_shist'])
        ax[1].plot(self.optimizer.stats_nat['mean_shist'])
        ax[1].set_xlabel('Generation cycles')
        ax[1].set_ylabel('Avg Target Activations')
        ax[1].set_title('... and Kreimann')
        ax[1].legend()


        save_figure_fp = path.join(save_dir_fp, f'scores.png')
        self._logger.info(mess=f"Saving figure to {save_figure_fp}")
        plt.savefig(save_figure_fp)
        
        save_image = make_grid([*self._best_img['gen'][0], *self._best_img['nat'][0]], nrow=1)
        save_image = cast(Image.Image, to_pil_image(save_image))
                
        save_img_fp = path.join(save_dir_fp, f'best_images.png')
        
        self._logger.info(mess=f"Saving best image to {save_img_fp}")

        save_image.save(save_img_fp)
        
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        self._stimuli, _ = data
        return super()._stimuli_to_sbj_state(data)
        
    def _sbj_state_to_sbj_score(self, data: Tuple[SubjectState, Message]) -> Tuple[StimuliScore, Message]:
        
        sbj_state, _ = data
        
        for k, v in sbj_state.items():
            self._rec_dict[k].append(v)
        
        return super()._sbj_state_to_stm_score(data)
    
    def _sbj_score_to_codes(self, data: Tuple[StimuliScore, Message]) -> Codes:
        sub_score, msg =data
        for imtype, mask in zip(['nat', 'gen'], [msg.mask, ~msg.mask]):
            
            max_, armax = tuple(f_func(sub_score[mask]) for f_func in [np.amax, np.argmax])
        
            if max_ > self._best[imtype]:
                self._best[imtype] = max_
                self._best_img[imtype] = self._stimuli[torch.tensor(mask)][armax]
        
        return super()._stm_score_to_codes((sub_score, msg))

    

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
    #get relevant input from args and do the necessary transformations
    population_size = args.pop_sz
    num_gens = args.num_gens
    base_seq = [char == 't' for char in args.mask_template]
    
    sbj_net = NetworkSubject(network_name = 'alexnet')
    l_names = sbj_net.layer_names
    rec_layers = [l_names[i] for i in args.rec_layers]
    
    score_dict = eval(args.score_layers)
    score_dict = {l_names[k]: v[1] if isinstance(v[1], list) 
                else random.sample(range(1000), v[0])
                for k, v in score_dict.items()}
    
    #define your generator
    mini_IN = _MiniImageNet(root=args.tiny_inet_root)
    mini_IN_loader = DataLoader(mini_IN, batch_size=2, shuffle=True)
    
    generator = InverseAlexGenerator(
        root=args.gen_root,
        variant=args.gen_variant,
        output_pipe=transform,
        nat_img_loader = mini_IN_loader
        ).to(device)
    
    #initialize the network subject (alexnet) with a recording probe
    record_dict = {l: None for l in rec_layers} #layers you want to record from
    my_probe = RecordingProbe(target = record_dict)
    sbj_net = NetworkSubject(record_probe = my_probe, network_name = 'alexnet')
    sbj_net._network.eval()
    
    #define a scorer
    scorer = MaxActivityScorer(trg_neurons = score_dict, aggregate=lambda x: np.mean(np.stack(list(x.values())), axis=0))

    #initialize the optimizer
    optim = GeneticOptimizer(
        states_shape=generator.input_dim,
        random_state=args.optimizer_seed,
        random_distr='normal',
        mutation_rate=args.mutation_rate,
        mutation_size=args.mutation_size,
        population_size=population_size,
        temperature=args.temperature,
        num_parents=args.num_parents)
    
    mask_generator = partial(repeat_pattern, base_seq = base_seq, shuffle = args.mask_is_random)
    
    data = {
    "save_dir": args.save_dir
    }
    
    experiment_config = ExperimentConfig(
        generator=generator,
        scorer=scorer,
        optimizer=optim,
        subject=sbj_net,
        logger=_LoguruLogger(),
        iteration=args.num_gens,
        mask_generator=mask_generator,
        data=data
    )
    
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
    
    parser.add_argument('-pop_sz',         type=int,   default=20,                      help='Number of images per generation')
    parser.add_argument('-num_gens',       type=int,   default=250,                     help='Number of total generations to evolve')
    parser.add_argument('-img_size',       type=tuple, default=(256, 256),              help='Size of a given image', nargs=2)
    parser.add_argument('-gen_variant',    type=str,   default='fc8',                   help='Variant of InverseAlexGenerator to use')
    parser.add_argument('-optimizer_seed', type=int,   default=31415,                   help='Random seed in GeneticOptimizer')
    parser.add_argument('-mutation_rate',  type=float, default=0.3,                     help='Mutation rate in GeneticOptimizer')
    parser.add_argument('-mutation_size',  type=float, default=0.3,                     help='Mutation size in GeneticOptimizer')
    parser.add_argument('-num_parents',    type=int,   default=2,                       help='Number of parents in GeneticOptimizer')
    parser.add_argument('-temperature',    type=float, default=1.0,                     help='Temperature in GeneticOptimizer')
    parser.add_argument('-mask_template',  type=str ,  default='tffffff',               help='String of True(t) and False(f). It will be converted in the basic sequence of the mask')
    parser.add_argument('-mask_is_random', type=bool , default=False,                   help='Defines if the mask is pseudorandom or not')
    parser.add_argument('-rec_layers',     type=tuple, default=(18,20),                 help='Layers you want to record from (each int in tuple = nr of the layer)')
    parser.add_argument('-score_layers',   type=str,   default="{20:(1,range(1))}",     help='Layers you want to score from (dictionary with layer nr: (nr of units, units ID (r = rand)))')

    parser.add_argument('-gen_root',       type=str,   default=gen_root,                help='Path to root folder of generator checkpoints')
    parser.add_argument('-tiny_inet_root', type=str,   default=tiny_inet_root,          help='Path to tiny imagenet dataset')
    parser.add_argument('-save_dir',       type=str,   default=image_out,               help='Path to store best solution')
    
    args = parser.parse_args()
    
    main(args)
    
