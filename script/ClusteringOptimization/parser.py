from argparse import ArgumentParser
from os import path

from zdream.utils.io_ import read_json


SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')


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
    cluster      = script_settings['clustering']
    config_path  = path.join(script_settings['config'], 'clustering_optimization.json')

    parser = ArgumentParser()
    
    # Configuration file
    parser.add_argument('--config',            type=def_type(str),   help='Path for the JSON configuration file', default = config_path)
    
    # Clustering
    parser.add_argument('--cluster_file',      type=def_type(str),   help='Path to clustering JSON file', default = cluster)
    parser.add_argument('--cluster_idx',       type=def_type(int),   help='Cluster index to optimize for')
    parser.add_argument('--weighted_score',    type=def_type(bool),  help='If to weight score by cluster rank')
    parser.add_argument('--layer',             type=def_type(str),   help='Layer name for which clustering was computed')
    parser.add_argument('--scr_type',          type=def_type(str),   help='Scoring units strategy {`cluster`; `random`; `random_adj`}')
    
    # Generator
    parser.add_argument('--variant',           type=def_type(str),   help='Variant of InverseAlexGenerator to use')
    parser.add_argument('--weights',           type=def_type(str),   help='Path to folder with generator weights', default = gen_weights)
    parser.add_argument('--mini_inet',         type=def_type(str),   help='Path to mini-imagenet dataset', default = mini_inet)
    parser.add_argument('--batch_size',        type=def_type(int),   help='Natural image dataloader batch size')
    
    # Mask generator
    parser.add_argument('--template',          type=def_type(str),   help='String of True(`T`) and False(`F`) as the basic sequence of the mask')
    parser.add_argument('--shuffle',           type=def_type(bool),  help='If to shuffle mask template')

    # Subject
    parser.add_argument('--net_name',          type=def_type(str),   help='SubjectNetwork name')

    # Scorer
    parser.add_argument('--layer_reduction',   type=def_type(str),   help='Name of reducing function across layers')
    
    # Optimizer
    parser.add_argument('--optimizer_type',    type=def_type(str),   help='Type of optimizer {{`genetic`, `cmaes`}}')
    parser.add_argument('--random_distr',      type=def_type(str),   help='Type of sampling random distribution')
    parser.add_argument('--random_scale',      type=def_type(float), help='Scale for random distribution')
    parser.add_argument('--pop_size',          type=def_type(int),   help='Starting number of the population')
    parser.add_argument('--mutation_rate',     type=def_type(float), help='Mutation rate for the optimizer')
    parser.add_argument('--mutation_size',     type=def_type(float), help='Mutation size for the optimizer')
    parser.add_argument('--n_parents',       type=def_type(int),   help='Number of parents for the optimizer')
    parser.add_argument('--topk',              type=def_type(int),   help='Number of codes of previous generation to keep')
    parser.add_argument('--temperature',       type=def_type(float), help='Temperature for the optimizer')
    parser.add_argument('--temperature_scale', type=def_type(float), help='Temperature scale per iteration')
    parser.add_argument('--sigma0',            type=def_type(float), help='Variance for CMAES optimizer')
    
    # Logger
    parser.add_argument('--name',              type=def_type(str),   help='Experiment name')
    parser.add_argument('--version',           type=def_type(int),   help='Experiment version')
    parser.add_argument('--out_dir',           type=def_type(str),   help='Path to directory to save outputs', default = out_dir)

    # Globals
    parser.add_argument('--iter',              type=def_type(int),   help='Number of total iterations')
    parser.add_argument('--random_seed',       type=def_type(int),   help='Random state for the experiment')
    parser.add_argument('--render',            type=def_type(bool),  help='If to render stimuli')

    return parser