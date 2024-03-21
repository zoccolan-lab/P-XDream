from argparse import ArgumentParser
from os import path

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
    out_dir  = script_settings['out_dir']
    inet_dir = script_settings['mini_inet']
    config_path  = script_settings['neural_recording_config']

    parser = ArgumentParser()
    
    # Configuration file
    parser.add_argument('--config',     type=def_type(str),   help='Path for the JSON configuration file', default = config_path)
        
    # Subject
    parser.add_argument('--net_name',   type=def_type(str),   help='Network name')
    parser.add_argument('--rec_layers', type=def_type(str),   help=f"Layers to record. {LAYERS_NEURONS_SPECIFICATION}")

    # Dataset
    parser.add_argument('--mini_inet',  type=def_type(str),   help='Path to Mini-Imagenet dataset', default=inet_dir)
    parser.add_argument('--image_ids',  type=def_type(str),   help='Image indexes for recording separated by a comma')

    # Logger
    parser.add_argument('--name',       type=def_type(str),   help='Experiment name')
    parser.add_argument('--version',    type=def_type(int),   help='Experiment version')
    parser.add_argument('--out_dir',    type=def_type(str),   help='Path to directory to save outputs', default = out_dir)
    
    # Globals
    parser.add_argument('--log_chk',    type=def_type(int),   help='Logger iteration checkpoint')

    return parser