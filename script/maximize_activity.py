"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib

from zdream.utils.experiment_types import _MaximizeActivityExperiment
from zdream.utils.misc import overwrite_dict, read_json

matplotlib.use('TKAgg')


LOCAL_SETTINGS = 'local_settings.json'

def main(args):    

    # Experiment

    json_conf = read_json(args.config)
    args_conf = {k : v for k, v in vars(args).items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf)
    
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
    mini_inet    = script_settings['mini_inet']
    config_path  = script_settings['maximize_activity_config']

    parser = ArgumentParser()
    
    parser.add_argument('--config',          type=str,   help='Path for the JSON configuration file',   default = config_path,)
    
    # Generator
    parser.add_argument('--weights',        type=str,   help='Path to folder of generator weights',     default = gen_weights,)
    parser.add_argument('--mini_inet',      type=str,   help='Path to mini mini imagenet dataset',      default = mini_inet,)
    parser.add_argument('--batch_size',     type=int,   help='Natural image dataloader batch size')
    parser.add_argument('--variant',        type=str,   help='Variant of InverseAlexGenerator to use')
    
    # Mask generator
    parser.add_argument('--template',       type=str ,  help='String of True(T) and False(F) as the basic sequence of the mask')
    parser.add_argument('--shuffle',        type=bool , help='If to shuffle mask pattern')

    # Subject
    parser.add_argument('--net_name',       type=str,   help='SubjectNetwork name')
    parser.add_argument('--rec_layers',     type=tuple, help='Recording layers')

    # Scorer
    parser.add_argument('--targets',        type=str, help='Target scoring layers and neurons')
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
