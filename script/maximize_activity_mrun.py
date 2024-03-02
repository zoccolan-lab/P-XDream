"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib
from zdream.experiment import MultiExperiment

from zdream.utils.experiment_types import _MaximizeActivityExperiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict, flatten_dict
from typing import cast

matplotlib.use('TKAgg')

LOCAL_SETTINGS = 'local_settings.json'

def main(args):   

    # Filter out None i.e. input not given
    args = {k: v for k, v in args.items() if v} 

    # Load default configuration
    json_conf = read_json(args['config'])

    # Get type dictionary for casting
    dict_type = {k: type(v) for k, v in flatten_dict(json_conf).items()}

    # Config from command line
    args_conf = {}

    # Keep track of argument lengths
    observed_lens = set()

    # Loop on input arguments
    for k, arg in args.items():

        # Get typing for cast
        type_cast = dict_type[k]

        # Split input line with separator # and cast
        args_conf[k] = [
            type_cast(a.strip()) for a in arg.split('#')
        ]

        # Add observed length if different from one
        n_arg = len(args_conf[k])
        if n_arg != 1:
            observed_lens.add(n_arg)

    # Check if multiple lengths
    if len(observed_lens) > 1:
        raise SyntaxError(f'Multiple argument with different lengths: {observed_lens}')
    
    # Check for no multiple args specified
    if len(observed_lens) == 0:
        raise SyntaxError(f'No multiple argument was specified.')
    
    # Adjust 1-length values
    n_args = list(observed_lens)[0]
    args_conf = {k : v * n_args if len(v) == 1 else v for k, v in args_conf.items()}

    print(args_conf)

    mrun_experiment = MultiExperiment(
        experiment=_MaximizeActivityExperiment,
        base_config=json_conf,
        search_config=args_conf
    )

    mrun_experiment.run()



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
    
    parser.add_argument('--config',         type=str,   help='Path for the JSON configuration file',   default = config_path,)
    
    # Generator
    parser.add_argument('--weights',        type=str,   help='Path to folder of generator weights',    default = gen_weights,)
    parser.add_argument('--mini_inet',      type=str,   help='Path to mini mini imagenet dataset',     default = mini_inet,)
    parser.add_argument('--batch_size',     type=str,   help='Natural image dataloader batch size')
    parser.add_argument('--variant',        type=str,   help='Variant of InverseAlexGenerator to use')
    
    # Mask generator
    parser.add_argument('--template',       type=str ,  help='String of True(T) and False(F) as the basic sequence of the mask')
    parser.add_argument('--shuffle',        type=bool , help='If to shuffle mask pattern')

    # Subject
    parser.add_argument('--net_name',       type=str,   help='SubjectNetwork name')
    parser.add_argument('--rec_layers',     type=str,   help='Recording layers')

    # Scorer
    parser.add_argument('--targets',        type=str,   help='Target scoring layers and neurons')
    parser.add_argument('--aggregation',    type=str, help='Name of scoring aggregation function between layers')
    parser.add_argument('--scr_rseed',      type=str, help='Random seed for neurons selection')
    
    # Optimizer
    parser.add_argument('--pop_sz',         type=str,   help='Starting number of the population')
    parser.add_argument('--optim_rseed',    type=str,   help='Random seed in for the optimizer')
    parser.add_argument('--mutation_rate',  type=str,   help='Mutation rate for the optimizer')
    parser.add_argument('--mutation_size',  type=str,   help='Mutation size for the optimizer')
    parser.add_argument('--num_parents',    type=str,   help='Number of parents for the optimizer')
    parser.add_argument('--temperature',    type=str,   help='Temperature for the optimizer')
    parser.add_argument('--random_state',   type=str , help='Random state for the optimizer')
    
    # Logger
    parser.add_argument('--name',           type=str,   help='Experiment name')
    parser.add_argument('--version',        type=str,   help='Experiment version')
    parser.add_argument('--out_dir',        type=str,   help='Path to directory to save outputs',       default = out_dir,)
    
    # Iterations
    parser.add_argument('--num_gens',       type=str,   help='Number of total generations to evolve')
    
    conf = vars(parser.parse_args())

    main(conf)
