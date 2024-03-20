from argparse import ArgumentParser
from os import path

from zdream.clustering.model import NeuronalRecording, PairwiseSimilarity
from zdream.logger import LoguruLogger
from zdream.subject import NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.dataset import MiniImageNet
from zdream.utils.io_ import read_json
from zdream.utils.parsing import parse_recording

SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

LAYERS_NEURONS_SPECIFICATION = '''
TMP
'''

def main(args): 
    
    # Extract layers info
    layer_info = NetworkSubject(network_name=args['net_name']).layer_info
    
    # PROBE
    record_target = parse_recording(input_str=args['rec_layers'], net_info=layer_info)
    
    if len(record_target) > 1:
        raise NotImplementedError('Recording only supports one layer. ')
    
    probe = RecordingProbe(target=record_target) # type: ignore

    # SUBJECTS
    sbj_net = NetworkSubject(
        record_probe=probe,
        network_name=args['net_name']
    )
    sbj_net._network.eval()
    
    # DATASET
    dataset = MiniImageNet(root=args['mini_inet'])
    
    # LOGGER
    logger = LoguruLogger(
        conf={
            'out_dir': args['out_dir'],
            'title':   'NeuralRecording',
            'name':    args['name'],
            'version': args['version']
        }
    )
    
    # NEURAL RECORDING
    recording = NeuronalRecording(
        subject=sbj_net,
        dataset=dataset,
        indexes=conf['image_ids'],
        stimulus_post=lambda x: x['imgs'].unsqueeze(dim=0),
        state_post=lambda x: list(x.values())[0][0],
        logger=logger
    )
    
    recording.record(log_chk=10)
    
    recording.save(out_dir=logger.target_dir)
    
    # SIMILARITIES
    pw = PairwiseSimilarity(recordings=recording.recordings)
    
    pw.cosine_similarity.save(out_dir=logger.target_dir, logger=logger)


if __name__ == '__main__':    

    # Loading custom local settings to set as defaults
    local_folder       = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, LOCAL_SETTINGS)
    script_settings    = read_json(path=script_settings_fp)
    
    # Set paths as defaults
    out_dir  = script_settings['out_dir']
    inet_dir = script_settings['mini_inet']

    parser = ArgumentParser()
    
    # Subject
    parser.add_argument('--net_name',   type=str,   help='Network name',                                      default='alexnet')
    parser.add_argument('--rec_layers', type=str,   help=f"Layers to record. {LAYERS_NEURONS_SPECIFICATION}", default='21=[]')
    
    # Dataset
    parser.add_argument('--mini_inet',  type=str,   help='Path to Mini-Imagenet dataset',                     default=inet_dir)
    parser.add_argument('--image_ids',  type=list,  help='Image indexes for recording')
    
    # Logger
    parser.add_argument('--name',       type=str,   help='Experiment name',                                   default='recording')
    parser.add_argument('--version',    type=int,   help='Experiment version',                                default=0)
    parser.add_argument('--out_dir',    type=str,   help='Path to directory to save outputs',                 default = out_dir)
    
    # Globals
    parser.add_argument('--log_ckp',    type=int,   help='Logging checkpoints',                               default = 10)
    
    conf = vars(parser.parse_args())
    
    main(conf)