from argparse import ArgumentParser
from os import path
import time
from typing import Any, Dict, List, cast

from zdream.clustering.model import NeuronalRecording, PairwiseSimilarity
from zdream.experiment import Experiment
from zdream.generator import Generator
from zdream.logger import Logger, LoguruLogger
from zdream.message import Message
from zdream.optimizer import Optimizer
from zdream.scorer import Scorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.dataset import MiniImageNet
from zdream.utils.io_ import read_json, save_json
from zdream.utils.misc import flatten_dict, stringfy_time
from zdream.utils.model import MaskGenerator
from zdream.utils.parsing import parse_int_list, parse_recording

class NeuralRecordingExperiment(Experiment):
    
    EXPERIMENT_TITLE = "NeuronalRecording"
    
    def __init__(
        self, 
        neuronal_recording: NeuronalRecording, 
        logger: Logger,
        data: Dict[str, Any] = dict(),
        name: str = 'experiment'
    ) -> None:
        
        super().__init__(
            # NOTE: We set None unused elements in the experiment
            generator=None, # type: ignore
            scorer=None,    # type: ignore
            optimizer=None, # type: ignore
            subject=None,   # type: ignore
            iteration=0, 
            logger=logger,
            data=data,
            name=name
        )
        
        self._neuronal_recording = neuronal_recording
        
        self._log_chk = cast(int, data['log_chk'])
        
        
    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'NeuralRecordingExperiment':
        
        sbj_conf = conf['subject']
        dat_conf = conf['dataset']
        log_conf = conf['logger']
        
        # Extract layers info
        layer_info = NetworkSubject(network_name=sbj_conf['net_name']).layer_info
        
        # --- PROBE ---
        record_target = parse_recording(input_str=sbj_conf['rec_layers'], net_info=layer_info)
        
        if len(record_target) > 1:
            raise NotImplementedError(f'Recording only supported for one layer. Multiple found: {record_target.keys()}.')
        
        probe = RecordingProbe(target=record_target) # type: ignore

        # --- SUBJECTS ---
        sbj_net = NetworkSubject(
            record_probe=probe,
            network_name=sbj_conf['net_name']
        )
        sbj_net._network.eval()
        
        # --- DATASET ---
        dataset = MiniImageNet(root=dat_conf['mini_inet'])
        
        # --- LOGGER ---
        log_conf['title'] = NeuralRecordingExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(conf=log_conf)
        
        # --- NEURAL RECORDING ---
        indexes = parse_int_list(dat_conf['image_ids'])
        
        neuronal_recording = NeuronalRecording(
            subject=sbj_net,
            dataset=dataset,
            image_ids=indexes,
            stimulus_post=lambda x: x['imgs'].unsqueeze(dim=0),
            state_post=lambda x: list(x.values())[0][0],
            logger=logger
        )
        
        # --- DATA ---
        data = {
            'log_chk': conf['log_chk']
        }
        
        return NeuralRecordingExperiment(
            neuronal_recording=neuronal_recording,
            logger=logger,
            data=data
        )
    
    def _init(self) -> Message:

        # Create experiment directory
        self._logger.create_target_dir()

        # Save and log parameters
        if self._param_config:

            flat_dict = flatten_dict(self._param_config)
            
            # Log
            self._logger.info(f"")
            self._logger.info(mess=str(self))
            self._logger.info(f"Parameters:")

            max_key_len = max(len(key) for key in flat_dict.keys()) + 1 # for padding

            for k, v in flat_dict.items():
                k_ = f"{k}:"
                self._logger.info(f'{k_:<{max_key_len}}   {v}')

            # Save
            config_param_fp = path.join(self.target_dir, 'params.json')
            self._logger.info(f"Saving param configuration to: {config_param_fp}")
            save_json(data=self._param_config, path=config_param_fp)

        # Components
        self._logger.info(f"")
        self._logger.info(f"Components:")
        self._logger.info(mess=f'Recording: {self._neuronal_recording}')
        self._logger.info(mess=f'Scorer:    {self.scorer}')
        self._logger.info(f"")
        
                
        # We generate an initial message containing the start time
        msg = Message(
            start_time = time.time()
        )
        
        return msg
    
    def _run(self, msg: Message) -> Message:
        
        self._neuronal_recording.record(log_chk=self._log_chk)

        return msg
    
    def _finish(self, msg : Message) -> Message:
        
        # Log total elapsed time
        str_time = stringfy_time(sec=msg.elapsed_time)
        self._logger.info(mess=f"Experiment finished successfully. Elapsed time: {str_time} s.")
        self._logger.info(mess="")
        
        # Save recordings
        self._neuronal_recording.save(out_dir=self.target_dir)
        
        # Save similarities
        pw = PairwiseSimilarity(recordings=self._neuronal_recording.recordings)
        pw.cosine_similarity.save(out_dir=self.target_dir, logger=self._logger)

        return msg



def main(args): 
    
    # Extract layers info
    layer_info = NetworkSubject(network_name=args['net_name']).layer_info
    
    # PROBE
    record_target = parse_recording(input_str=args['rec_layers'], net_info=layer_info)
    
    if len(record_target) > 1:
        raise NotImplementedError('Recording only supports one layer. ')
    
    probe = RecordingProbe(target=record_target) # type: ignore
    