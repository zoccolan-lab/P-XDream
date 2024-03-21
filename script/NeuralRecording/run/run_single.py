from script.NeuralRecording.neural_recording import NeuralRecordingExperiment
from script.NeuralRecording.parser import get_parser
from zdream.utils.io_ import read_json
from zdream.utils.misc import overwrite_dict

def main(args): 
    
    # Experiment

    json_conf = read_json(args['config'])
    args_conf = {k : v for k, v in args.items() if v}
    
    full_conf = overwrite_dict(json_conf, args_conf)
        
    experiment = NeuralRecordingExperiment.from_config(full_conf)
    experiment.run()

if __name__ == '__main__':
    
    parser = get_parser()
    
    conf = vars(parser.parse_args())
    
    main(conf)
