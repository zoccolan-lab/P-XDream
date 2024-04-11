from argparse import ArgumentParser
from os import path

from zdream.utils.io_ import read_json


SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')


def get_parser(multirun: bool = False) -> ArgumentParser:
    '''
    Return the argument parser for DSClustering experiment.

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
    neural_recordings = script_settings['recordings']
    out_dir           = script_settings['out_dir']
    config_path       = path.join(script_settings['config'], 'ds_clustering.json')

    parser = ArgumentParser()
    
    # Configuration file
    parser.add_argument('--config',         type=def_type(str),   help='Path for the JSON configuration file', default = config_path)

    # Logger
    parser.add_argument('--name',           type=def_type(str),   help='Experiment name')
    parser.add_argument('--version',        type=def_type(int),   help='Experiment version')
    parser.add_argument('--out_dir',        type=def_type(str),   help='Path to directory to save outputs', default = out_dir)
    
    # Clustering
    parser.add_argument('--recordings',      type=def_type(str),   help='Path to neural recordings file', default=neural_recordings)
    parser.add_argument('--max_iter',        type=def_type(int),   help='Maximum number of iterations')
    parser.add_argument('--min_elements',    type=def_type(int),   help='Minimum cluster cardinality')
    
    return parser