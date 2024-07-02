from experiment.utils.cmdline_args import Args
from experiment.utils.misc import run_single
from experiment.NeuralRecording.args import ARGS
from experiment.NeuralRecording.neural_recording import NeuralRecordingExperiment


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=NeuralRecordingExperiment)

