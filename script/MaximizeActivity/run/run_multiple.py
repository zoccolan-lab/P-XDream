"""
TODO Experiment description
"""

from os import path
from argparse import ArgumentParser
import matplotlib
import tkinter as tk


from script.MaximizeActivity.maximize_activity import NeuronScalingMultiExperiment, LayersCorrelationMultiExperiment, MaximizeActivityExperiment
from script.MaximizeActivity.parser import get_parser
from zdream.experiment import MultiExperiment
from zdream.utils.io_ import read_json
from zdream.utils.misc import flatten_dict

matplotlib.use('TKAgg')

# NOTE: Script directory path refers to the current script file


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
    
    mrun_experiment = LayersCorrelationMultiExperiment(
        experiment=MaximizeActivityExperiment,
        default_conf=json_conf,
        experiment_conf=args_conf
    )

    mrun_experiment.run()



if __name__ == '__main__':
    
    parser = get_parser(multirun=True)
    
    conf = vars(parser.parse_args())

    main(conf)
