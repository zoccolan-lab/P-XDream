from script.cmdline_args import Args
from script.script_utils import run_single
from script.NeuralRecording.args import ARGS
from script.NeuralRecording.neural_recording import NeuralRecordingExperiment


if __name__ == '__main__':

    parser = Args.get_parser(args=ARGS)
    args = vars(parser.parse_args())
    
    run_single(args=args, exp_type=NeuralRecordingExperiment)

