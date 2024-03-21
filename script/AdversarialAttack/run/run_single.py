"""
TODO Experiment description
"""

import matplotlib
from os import path

from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict
from zdream.utils.model import DisplayScreen
from script.AdversarialAttack.parser import get_parser
from script.AdversarialAttack.adversarial_attack import AdversarialAttackExperiment

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
    
    experiment = AdversarialAttackExperiment.from_config(full_conf)
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
    config_path  = script_settings['adversarial_attack_config']

    parser = get_parser()
    
    conf = vars(parser.parse_args())
    
    main(conf)
