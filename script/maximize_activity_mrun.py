"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib
from zdream.experiment import MultiExperiment

from zdream.utils.experiment_types import _MaximizeActivityExperiment
from zdream.utils.misc import overwrite_dict, read_json, flatten_dict
from typing import cast

matplotlib.use('TKAgg')

LOCAL_SETTINGS = 'local_settings.json'

def main(args):    

    # Experiment

    json_conf = read_json(args.config)
    types_json_dict = flatten_dict(json_conf, get_type=True)

    args_conf = {}; n_items = None
    for k,v in vars(args).items():
        if isinstance(v, str) and not('/' in v):
            v_list = v.split('#')
            v_type = types_json_dict[k]
            args_conf[k] = [v_type(v) for v in v_list]
            if n_items and not(n_items == len(v_list) or len(v_list)==1):
                raise ValueError('inputs of different lengths')
            
            if len(v_list) > 1:
                n_items = len(v_list) 
                
    
    args_conf = {k : [v]*cast(int,n_items) for k, v in args_conf.items() if (v and len(v) == 1)}

            
    print(args_conf)

    #args_conf = {k : [v]*3 for k, v in vars(args).items() if v}
    #args_conf['num_gens'] = (2, 3, 4)

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
    parser.add_argument('--weights',        type=str,   help='Path to folder of generator weights',     default = gen_weights,)
    parser.add_argument('--mini_inet',      type=str,   help='Path to mini mini imagenet dataset',      default = mini_inet,)
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
    parser.add_argument('--aggregation',    type=tuple, help='Name of scoring aggregation function between layers')
    parser.add_argument('--scr_rseed',      type=tuple, help='Random seed for neurons selection')
    
    # Optimizer
    parser.add_argument('--pop_sz',         type=str,   help='Starting number of the population')
    parser.add_argument('--optim_rseed',    type=str,   help='Random seed in for the optimizer')
    parser.add_argument('--mutation_rate',  type=str,   help='Mutation rate for the optimizer')
    parser.add_argument('--mutation_size',  type=str,   help='Mutation size for the optimizer')
    parser.add_argument('--num_parents',    type=str,   help='Number of parents for the optimizer')
    parser.add_argument('--temperature',    type=str,   help='Temperature for the optimizer')
    parser.add_argument('--random_state',   type=bool , help='Random state for the optimizer')
    
    # Logger
    parser.add_argument('--name',           type=str,   help='Experiment name')
    parser.add_argument('--version',        type=int,   help='Experiment version')
    parser.add_argument('--out_dir',        type=str,   help='Path to directory to save outputs',       default = out_dir,)
    
    # Iterations
    parser.add_argument('--num_gens',       type=str,   help='Number of total generations to evolve')
    
    conf = parser.parse_args()
    main(conf)
