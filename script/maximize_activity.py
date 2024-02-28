"""
TODO Experiment description
"""

import os
from os import path
import numpy as np
from argparse import ArgumentParser, Namespace
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from numpy.typing import NDArray
from typing import Any, Dict, List, cast, Tuple
from functools import partial
import random
import matplotlib
from PIL import Image
import pickle

from zdream.logger import LoguruLogger
from zdream.generator import InverseAlexGenerator
from zdream.plot_utils import plot_optimization_profile, plot_scores_by_cat
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject
from zdream.scores import MaxActivityScorer
from zdream.optimizer import GeneticOptimizer
from zdream.experiment import Experiment, ExperimentConfig
from zdream.utils.dataset import MiniImageNet
from zdream.utils.misc import concatenate_images, read_json, repeat_pattern, device, to_gif
from zdream.utils.model import Codes, Message, Stimuli, StimuliScore, SubjectState

matplotlib.use('TKAgg')


class _MaximizeActivityExperiment(Experiment):

    _EXPERIMENT_NAME = "MaximizeActivity"

    @classmethod
    def from_config(cls, conf : Dict[str, Any]) -> '_MaximizeActivityExperiment':
        '''
        Static constructor for a _MaximizeActivityExperiment class from configuration file.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: _MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: _MaximizeActivity
        '''

        # Transform Mask sequence into boolean
        base_seq = [char == 't' for char in conf['mask_template']] # TODO It doesn't check for others letter instead of 'f'
        
        # Create network subject
        sbj_net = NetworkSubject(network_name='alexnet')
        
        layer_names = sbj_net.layer_names
        
        # Extract the name of recording layers
        rec_layers = [layer_names[i] for i in conf['rec_layers']] # TODO Check input as a tuple

        # Building target neurons
        # TODO what is this ifelse doing?
        score_dict = {}
        random.seed(conf['score_rseed'])
        for sl, su in zip(conf['score_layers'], conf['score_units']):
            if isinstance(su,list):
                k_vals = list(range(su[0], su[1]))
            else:
                k_vals = random.sample(range(1000),su)
            score_dict[layer_names[sl]] = k_vals
            
        # Generator with Dataloader
        mini_IN = MiniImageNet(root=conf['tiny_inet_root'])
        mini_IN_loader = DataLoader(mini_IN, batch_size=10, shuffle=True) #set num_workers
        # mini_IN_loader = DataLoader(RandomImageDataset(1000, (3, 256, 256)), batch_size=2, shuffle=True)
        
        generator = InverseAlexGenerator(
            root=conf['gen_root'],
            variant=conf['gen_variant'],
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
            random_state=conf['optimizer_seed'],
            random_distr='normal',
            mutation_rate=conf['mutation_rate'],
            mutation_size=conf['mutation_size'],
            population_size=conf['pop_sz'],
            temperature=conf['temperature'],
            num_parents=conf['num_parents']
        )

        # Logger
        logger = LoguruLogger(
            out_dir=conf['save_dir'],
            exp_name=_MaximizeActivityExperiment._EXPERIMENT_NAME,
            exp_version=conf['exp_vrs']
        )
        
        # Mask generator
        mask_generator = partial(repeat_pattern, base_seq=base_seq, shuffle=conf['mask_is_random'])
        
        # Additional data
        data = {
            "save_dir": conf['save_dir']
        }
        
        # Experiment configuration
        experiment_config = ExperimentConfig(
            generator=generator,
            scorer=scorer,
            optimizer=optim,
            subject=sbj_net,
            logger=logger,
            iteration=conf['num_gens'],
            mask_generator=mask_generator,
            data=data
        )
        
        experiment = cls(experiment_config, version = conf['exp_vrs'])

        experiment._set_param_configuration(param_config=conf)

        return experiment

    def __init__(self, config: ExperimentConfig, version : str, name: str = "maximize_activity") -> None:
        
        super().__init__(config=config, version=name)
        self._data = cast(Dict[str, Any], config.data)
        
    def _progress_info(self, i: int) -> str:
        
        # We add the progress information about the best
        # and the average score per each iteration
        stat_gen = self.optimizer.stats
        stat_nat = self.optimizer.stats_nat
        
        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()
        best_nat = cast(NDArray, stat_nat['best_score']).mean()

        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        best_nat_str = f'{best_nat:.1f}'
        
        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str} | best nat: {best_nat_str}'
        
        progress_super = super()._progress_info(i=i)
        
        return f'{progress_super}{desc}'
        
    def _init(self):
        
        super()._init()
        
        # Data structure to save best score and best image
        self._best_scr = {'gen': 0, 'nat': 0}
        self._best_img = {
            k: torch.zeros(self.generator.output_dim, device = device)
            for k in ['gen', 'nat']
        }
        
        # Set screen
        self._screen_syn = "Best synthetic image"
        self._logger.add_screen(screen_name=self._screen_syn, display_size=(400,400))
        
        self._screen_nat = "Best natural image"
        self._logger.add_screen(screen_name=self._screen_nat, display_size=(400,400))

        # Set gif
        self._gif: List[Image.Image] = []

        # Last seen labels
        self._labels: List[int] = []

        
    def _progress(self, i: int):
        
        super()._progress(i)
        
        # Get best stimuli
        best_code = self.optimizer.solution
        best_synthetic, _ = self.generator(best_code)
        best_synthetic_img = to_pil_image(best_synthetic[0])

        best_natural = self._best_img['nat']
        
        self._logger.update_screen(
            screen_name=self._screen_syn,
            image=best_synthetic_img
        )

        self._logger.update_screen(
            screen_name=self._screen_nat,
            image=to_pil_image(best_natural)
        )

        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )
        
    def _finish(self):
        
        super()._finish()
        
        # Close screens
        self._logger.remove_all_screens()
        
        # 1) Best Images
        
        # We create the directory to store results
        out_dir_fp = path.join(self._data['save_dir'], self._version, self._version)
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
        out_fp = path.join(self.target_dir, f'best_images.png')
        self._logger.info(mess=f"Saving best images to {out_fp}")
        out_image.save(out_fp)

        # Gif
        out_gif_fp = path.join(self.target_dir, f'best_image.gif')
        self._logger.info(mess=f"Saving best image gif to {out_gif_fp}")
        to_gif(image_list=self._gif, out_fp=out_gif_fp)
        
        # 2) Score plots
        plot_optimization_profile(self._optimizer, save_dir = self.target_dir)
        plot_scores_by_cat(self._optimizer, self._labels, save_dir = self.target_dir)
        
        # TODO pkl_name = '_'.join([self._version, self._version])+'.pkl'
        # TODO with open(path.join(out_dir_fp,pkl_name), 'wb') as file:
        # TODO     pickle.dump(self, file)
        
    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:
        
        # We save the last set of stimuli
        self._stimuli, msg = data
        self._labels.extend(msg.label)
        
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

def main(args):    
    # Experiment
    
    json_conf = read_json(args.config)
    args_conf = {k : v for k, v in vars(args).items() if v}
    
    full_conf = {**json_conf, **args_conf}
    
    print(f'Launching MaximizeActivity Experiment with the following configuration:\n {full_conf}')
    
    experiment = _MaximizeActivityExperiment.from_config(full_conf)
    experiment.run()


if __name__ == '__main__':
    
    # Loading `local_settings.json` for custom local settings
    local_folder = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, 'local_settings.json')
    script_settings = read_json(path=script_settings_fp)
    
    gen_root   = script_settings['inverse_alex_net']
    image_out  = script_settings['image_out']
    tiny_inet_root = script_settings['tiny_inet_root']
    config_path = script_settings['config_file_maximize_activity']

    parser = ArgumentParser()
    
    parser.add_argument('-config',          type=str, default = config_path,      help='Path for the JSON configuration file')
    parser.add_argument('-gen_root',        type=str, default = gen_root,         help='Path to root folder of generator checkpoints')
    parser.add_argument('-tiny_inet_root',  type=str, default = tiny_inet_root,   help='Path to tiny imagenet dataset')
    parser.add_argument('-save_dir',        type=str, default = image_out,        help='Path to store data')
    
    parser.add_argument('--exp_vrs',        type=str,   help='Experiment version name')
    parser.add_argument('--pop_sz',         type=int,   help='Number of images per generation')
    parser.add_argument('--num_gens',       type=int,   help='Number of total generations to evolve')
    parser.add_argument('--img_size',       type=tuple, help='Size of a given image', nargs=2)
    parser.add_argument('--gen_variant',    type=str,   help='Variant of InverseAlexGenerator to use')
    parser.add_argument('--optimizer_seed', type=int,   help='Random seed in GeneticOptimizer')
    parser.add_argument('--mutation_rate',  type=float, help='Mutation rate in GeneticOptimizer')
    parser.add_argument('--mutation_size',  type=float, help='Mutation size in GeneticOptimizer')
    parser.add_argument('--num_parents',    type=int,   help='Number of parents in GeneticOptimizer')
    parser.add_argument('--temperature',    type=float, help='Temperature in GeneticOptimizer')
    parser.add_argument('--mask_template',  type=str ,  help='String of True(t) and False(f). It will be converted in the basic sequence of the mask')
    parser.add_argument('--mask_is_random', type=bool , help='Defines if the mask is pseudorandom or not')
    parser.add_argument('--rec_layers',     type=tuple, help='Layers you want to record from (each int in tuple = nr of the layer)')

    parser.add_argument('--score_layers',    type=tuple, help='Layers you want to score from')
    parser.add_argument('--score_units',     type=tuple, help='Units you want to score from')
    parser.add_argument('--score_rseed',     type=tuple, help='random seed for selecting units')
    
    conf = parser.parse_args()
    
    main(conf)
    
