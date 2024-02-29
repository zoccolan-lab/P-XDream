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

from zdream.logger import LoguruLogger
from zdream.generator import InverseAlexGenerator
from zdream.utils.plotting import plot_optimization_profile, plot_scores_by_cat
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject
from zdream.scores import MaxActivityScorer
from zdream.optimizer import GeneticOptimizer
from zdream.experiment import Experiment, ExperimentConfig
from zdream.utils.dataset import MiniImageNet
from zdream.utils.misc import concatenate_images, merge_dicts, overwrite_dict, read_json, repeat_pattern, device, to_gif
from zdream.utils.model import Codes, Message, Stimuli, StimuliScore, SubjectState, aggregating_functions

matplotlib.use('TKAgg')


class _MaximizeActivityExperiment(Experiment):

    _EXPERIMENT_NAME = "MaximizeActivity"

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> '_MaximizeActivityExperiment':
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

        # --- GENERATOR ---

        # Dataloader
        dataset    = MiniImageNet(root=gen_conf['tiny_inet'])
        dataloader = DataLoader(dataset, batch_size=gen_conf['batch_size'], shuffle=True)
        
        # Instance
        generator = InverseAlexGenerator(
            root           = gen_conf['weights'],
            variant        = gen_conf['variant'],
            nat_img_loader = dataloader
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_names: List[str] = NetworkSubject(network_name=sbj_conf['net_name']).layer_names

        
        # Probe
        record_target = {layer_names[i]: None for i in sbj_conf['rec_layers']} # Record from any neurons in input layers
        probe = RecordingProbe(target = record_target) # type: ignore
        
        sbj_net = NetworkSubject(
            record_probe=probe, 
            network_name=sbj_conf['net_name']
        )
        sbj_net._network.eval() # TODO cannot access private attribute, make public method to call the eval
        
        # --- SCORER ---

        # Target neurons
        score_dict = {}
        random.seed(scr_conf['scr_rseed']) # TODO Move to numpy random

        for layer, units in zip(scr_conf['target_layers'], scr_conf['target_units']):

            # In the case the units are a list of length two we take
            # the neurons in their range
            if isinstance(units, list):
                if len(units) == 2:
                    start, end = units
                    neurons = list(range(start, end))
                else:
                    raise ValueError('Units provided must be one or two')
            # In the case the units are a single number 
            # we uniformly sample that number of neurons
            elif isinstance(units, int):
                neurons = random.sample(range(np.prod(generator.input_dim)), units)
            else:
                raise TypeError('Units must be a list of length two or integers.')
            
            # Add layer and neurons to scoring
            score_dict[layer_names[layer]] = neurons

        scorer = MaxActivityScorer(
            trg_neurons=score_dict,
            aggregate=aggregating_functions[scr_conf['aggregation']]
        )

        # --- OPTIMIZER ---

        optim = GeneticOptimizer(
            states_shape   = generator.input_dim,
            random_state   = opt_conf['optim_rseed'],
            random_distr   = opt_conf['random_state'],
            mutation_rate  = opt_conf['mutation_rate'],
            mutation_size  = opt_conf['mutation_size'],
            population_size= opt_conf['pop_sz'],
            temperature    = opt_conf['temperature'],
            num_parents    = opt_conf['num_parents']
        )

        # Logger
        log_conf['title'] = _MaximizeActivityExperiment._EXPERIMENT_NAME
        logger = LoguruLogger(
            log_conf=log_conf
        )
        
        # Mask generator
        base_seq = [char == 'T' for char in msk_conf['template']] 
        mask_generator = partial(repeat_pattern, base_seq=base_seq, shuffle=msk_conf['shuffle'])
        
        # Experiment configuration
        experiment_config = ExperimentConfig(
            generator=generator,
            scorer=scorer,
            optimizer=optim,
            subject=sbj_net,
            logger=logger,
            iteration=conf['num_gens'],
            mask_generator=mask_generator
        )
        
        experiment = cls(experiment_config, name=log_conf['name'])

        return experiment

        
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
        out_dir_fp = path.join(self.target_dir, self._name, self._name)
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


LOCAL_SETTINGS = 'local_settings.json'

def main(args):    

    # Experiment

    json_conf = read_json(args.config)
    args_conf = {k : v for k, v in vars(args).items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf)
    
    print(full_conf)
    
    experiment = _MaximizeActivityExperiment.from_config(full_conf)
    experiment.run()


if __name__ == '__main__':
    
    # Loading custom local settings
    local_folder       = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, LOCAL_SETTINGS)
    script_settings    = read_json(path=script_settings_fp)
    
    # Set as defaults
    gen_weights  = script_settings['gen_weights']
    out_dir      = script_settings['out_dir']
    tiny_inet    = script_settings['tiny_inet']
    config_path  = script_settings['maximize_activity_config']

    parser = ArgumentParser()
    
    parser.add_argument('--config',          type=str,   help='Path for the JSON configuration file',   default = config_path,)
    
    # Generator
    parser.add_argument('--weights',        type=str,   help='Path to folder of generator weights',     default = gen_weights,)
    parser.add_argument('--tiny_inet',      type=str,   help='Path to tiny tiny imagenet dataset',      default = tiny_inet,)
    parser.add_argument('--batch_size',     type=int,   help='Natural image dataloader batch size')
    parser.add_argument('--variant',        type=str,   help='Variant of InverseAlexGenerator to use')
    
    # Mask generator
    parser.add_argument('--template',       type=str ,  help='String of True(T) and False(F) as the basic sequence of the mask')
    parser.add_argument('--shuffle',        type=bool , help='If to shuffle mask pattern')

    # Subject
    parser.add_argument('--net_name',       type=str,   help='SubjectNetwork name')
    parser.add_argument('--rec_layers',     type=tuple, help='Recording layers')

    # Scorer
    parser.add_argument('--target_layers',  type=tuple, help='Target scoring layers')
    parser.add_argument('--target_units',   type=tuple, help='Target scoring neurons')
    parser.add_argument('--aggregation',    type=tuple, help='Name of scoring aggregation function between layers')
    parser.add_argument('--scr_rseed',      type=tuple, help='Random seed for neurons selection')
    
    # Optimizer
    parser.add_argument('--pop_sz',         type=int,   help='Starting number of the population')
    parser.add_argument('--optim_rseed',    type=int,   help='Random seed in for the optimizer')
    parser.add_argument('--mutation_rate',  type=float, help='Mutation rate for the optimizer')
    parser.add_argument('--mutation_size',  type=float, help='Mutation size for the optimizer')
    parser.add_argument('--num_parents',    type=int,   help='Number of parents for the optimizer')
    parser.add_argument('--temperature',    type=float, help='Temperature for the optimizer')
    parser.add_argument('--random_state',   type=bool , help='Random state for the optimizer')
    
    # Logger
    parser.add_argument('--name',           type=str,   help='Experiment name')
    parser.add_argument('--version',        type=int,   help='Experiment version')
    parser.add_argument('--out_dir',        type=str,   help='Path to directory to save outputs',       default = out_dir,)
    
    # Iterations
    parser.add_argument('--num_gens',       type=int,   help='Number of total generations to evolve')
    
    conf = parser.parse_args()
    
    main(conf)
