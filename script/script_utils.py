from typing import Any, Dict, Type
from zdream.experiment import Experiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict
from zdream.utils.model import DisplayScreen


def run_single(
    args: Dict[str, Any],
    exp_type:Type[Experiment]
):
    
    json_conf = read_json(args['config'])
    args_conf = {k : v for k, v in args.items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf) 
    
    # Hold main display screen reference
    if full_conf['render']:
        main_screen = DisplayScreen.set_main_screen()

    # Add close screen flag on as the experiment
    # only involves one run
    full_conf['close_screen'] = True
    
    experiment = exp_type.from_config(full_conf)
    experiment.run()