"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib

from script.MaximizeActivity.maximize_activity import _MaximizeActivityExperiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict

matplotlib.use('TKAgg')

# NOTE: Script directory path refers to the current script file
SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

def main(args):    

    # Experiment

    json_conf = read_json(args['config'])
    args_conf = {k : v for k, v in args.items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf)
    
    experiment = _MaximizeActivityExperiment.from_config(full_conf)
    experiment.run()

target_units_help = '''
The format to specify the structure is requires separating dictionaries with comma:
        layer1=units1, layer2=units2, ..., layerN=unitsN

    Where units specification can be:
        - all neurons; requires no specification:
            []
        - individual unit specification; requires neurons index to be separated by a space:
            [(A1 A2 A3) (B1 B2 B3) ...] <-- each neuron is identified by a tuple of numbers
            [A B C D ...]               <-- each neuron is identified by a single number
        - units from file; requires to specify a path to a .txt file containing one neuron per line
            (where each neuron is expected to be a tuple of ints separated by space or a single number):
            [file.txt]
        - A set of neurons in a given range:
            [A:B:step C:D:step E]
        - random set of N neurons:
            Nr[]
        - random set of N neuron, in a given range:
            Nr[A:B C:D E]
'''


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
    parser.add_argument('--mini_inet',      type=str,   help='Path to mini imagenet dataset',           default = mini_inet,)
    parser.add_argument('--batch_size',     type=int,   help='Natural image dataloader batch size')
    parser.add_argument('--variant',        type=str,   help='Variant of InverseAlexGenerator to use')
    
    # Mask generator
    parser.add_argument('--template',       type=str ,  help='String of True(T) and False(F) as the basic sequence of the mask')
    parser.add_argument('--shuffle',        type=bool , help='If to shuffle mask pattern')

    # Subject
    parser.add_argument('--net_name',       type=str,   help='SubjectNetwork name')
    parser.add_argument('--rec_layers',     type=str,   help='Recording layers')

    # Scorer
    parser.add_argument('--targets',        type=str,   help=target_units_help)
    parser.add_argument('--aggregation',    type=str,   help='Name of scoring aggregation function between layers')
    parser.add_argument('--scr_rseed',      type=str,   help='Random seed for neurons selection')
    
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

    # Output options
    parser.add_argument('--display_plots',  type=bool,  help='If to display plots')
    
    conf = vars(parser.parse_args())
    
    main(conf)
