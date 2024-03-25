from argparse import ArgumentParser
from functools import partial
from os import path
from typing import Type

from zdream.utils.io_ import read_json


SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

LAYERS_NEURONS_SPECIFICATION = '''
TMP
'''


def get_parser(multirun: bool = False) -> ArgumentParser:
    '''
    Return the argument parser for MaximizeActivity experiment.

    :param multirun: If the experiment is in multi-run version, indicating
                     all input arguments to be strings.
    :type multirun: bool, optional
    '''

    # We preserve the default type in the case of a multirun,
    # otherwise all arguments default to string.
    def_type = lambda x: str if multirun else x

    # Loading custom local settings to set as defaults
    local_folder       = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, LOCAL_SETTINGS)
    script_settings    = read_json(path=script_settings_fp)
    
    # Set paths as defaults
    gen_weights  = script_settings['gen_weights']
    out_dir      = script_settings['out_dir']
    mini_inet    = script_settings['mini_inet']
    config_path  = path.join(script_settings['config'], 'maximize_activity.json')

    parser = ArgumentParser()
    
    # Configuration file
    parser.add_argument('--config',         type=def_type(str),   help='Path for the JSON configuration file', default = config_path)
    
    # Generator
    parser.add_argument('--weights',        type=def_type(str),   help='Path to folder with generator weights', default = gen_weights,)
    parser.add_argument('--mini_inet',      type=def_type(str),   help='Path to mini-imagenet dataset', default = mini_inet,)
    parser.add_argument('--batch_size',     type=def_type(int),   help='Natural image dataloader batch size')
    parser.add_argument('--variant',        type=def_type(str),   help='Variant of InverseAlexGenerator to use')
    
    # Mask generator
    parser.add_argument('--template',       type=def_type(str),   help='String of True(T) and False(F) as the basic sequence of the mask')
    parser.add_argument('--shuffle',        type=def_type(bool),  help='If to shuffle mask template')

    # Subject
    parser.add_argument('--net_name',       type=def_type(str),   help='SubjectNetwork name')
    parser.add_argument('--rec_layers',     type=def_type(str),   help=f'Recording layers with specification\n{LAYERS_NEURONS_SPECIFICATION}')

    # Scorer
    parser.add_argument('--scr_layers',     type=def_type(str),   help=f'Target scoring layers and neurons with specification\n{LAYERS_NEURONS_SPECIFICATION}')
    parser.add_argument('--aggregation',    type=def_type(str),   help='Name of scoring aggregation function between layers')
    parser.add_argument('--reduction',      type=def_type(str),   help='Name of scoring aggregation function across layers')
    
    # Optimizer
    parser.add_argument('--pop_size',       type=def_type(int),   help='Starting number of the population')
    parser.add_argument('--mutation_rate',  type=def_type(float), help='Mutation rate for the optimizer')
    parser.add_argument('--mutation_size',  type=def_type(float), help='Mutation size for the optimizer')
    parser.add_argument('--num_parents',    type=def_type(int),   help='Number of parents for the optimizer')
    parser.add_argument('--topk',           type=def_type(int),   help='Number of codes of previous generation to keep')
    parser.add_argument('--temperature',    type=def_type(float), help='Temperature for the optimizer')
    
    # Logger
    parser.add_argument('--name',           type=def_type(str),   help='Experiment name')
    parser.add_argument('--version',        type=def_type(int),   help='Experiment version')
    parser.add_argument('--out_dir',        type=def_type(str),   help='Path to directory to save outputs', default = out_dir)
    
    # Globals
    parser.add_argument('--iter',           type=def_type(int),   help='Number of total iterations')
    parser.add_argument('--display_plots',  type=def_type(bool),  help='If to display plots')
    parser.add_argument('--random_seed',    type=def_type(int),   help='Random state for the experiment')
    parser.add_argument('--render',         type=def_type(bool),  help='If to render stimuli')

    return parser