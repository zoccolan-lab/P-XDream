"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib
import tkinter as tk

from script.MaximizeActivity.maximize_activity import MaximizeActivityExperiment
from script.MaximizeActivity.parser import get_parser
from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict
from zdream.utils.model import DisplayScreen

matplotlib.use('TKAgg')

# NOTE: Script directory path refers to the current script file
SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

def main(args): 
    
    # Experiment

    json_conf = read_json(args['config'])
    args_conf = {k : v for k, v in args.items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf) 
    
    # Hold main display screen reference
    if full_conf['render']:
        main_screen = DisplayScreen.set_main_screen()

    # Add close screen flag on as the experiment
    # only involves one run
    full_conf['close_screen'] = True
    
    experiment = MaximizeActivityExperiment.from_config(full_conf)
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

    parser = get_parser()
    
    conf = vars(parser.parse_args())
    
    main(conf)
