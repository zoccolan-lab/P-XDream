from script.NeuralRecording.neural_recording import NeuralRecordingExperiment
from script.NeuralRecording.parser import get_parser

def main(args):

    experiment = NeuralRecordingExperiment.from_args(args=args)
    experiment.run()

if __name__ == '__main__':
    
    parser = get_parser()
    
    args = vars(parser.parse_args())
    
    main(args=args)
