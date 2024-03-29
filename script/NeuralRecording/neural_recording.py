from os import path
import os
import time
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from torch.utils.data import Dataset

from zdream.experiment import Experiment
from zdream.logger import Logger, LoguruLogger, MutedLogger
from zdream.message import ZdreamMessage
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.probe import RecordingProbe
from zdream.utils.dataset import MiniImageNet
from zdream.utils.io_ import save_json
from zdream.utils.misc import device
from zdream.utils.parsing import parse_int_list, parse_recording
from zdream.message import Message

class NeuralRecordingExperiment(Experiment):
    
    EXPERIMENT_TITLE = "NeuralRecording"
    
    def __init__(
        self,
        subject   : InSilicoSubject,
        dataset   : Dataset,
        image_ids : List[int] = [],
        logger    : Logger = MutedLogger(),
        name      : str = 'neuronal_recording',
        data      : Dict[str, Any] = dict()
    ):
    
        super().__init__(
            name=name,
            logger=logger,
            data=data
        )
        
        if not image_ids:
            image_ids = list(range(len(dataset))) # type: ignore
        
        self._subject   = subject
        self._image_ids = image_ids
        self._dataset   = dataset
        self._log_chk   = cast(int, data['log_chk'])

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

        # --- SUBJECT ---
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
        image_ids = parse_int_list(dat_conf['image_ids'])
        
        # --- DATA ---
        data = {
            'log_chk': conf['log_chk']
        }
        
        return NeuralRecordingExperiment(
            subject   = sbj_net,
            dataset   = dataset,
            image_ids = image_ids,
            name      = log_conf['name'],
            logger    = logger,
            data      = data
        )
        
    @property
    def _components(self) -> List[Tuple[str, Any]]:
        return [
            ('Subject', self._subject),
            ('Dataset', self._dataset)
        ]
    
    
    # --- RUN ---

    def _run(self, msg: Message) -> Message:
        
        sbj_states = []
        
        # Post-processing operations
        stimuli_post = lambda x: x['imgs'].unsqueeze(dim=0)
        state_post   = lambda x: list(x.values())[0][0]        
        
        for i, idx in enumerate(self._image_ids):
            
            # Log progress
            if i % self._log_chk == 0:
                progress = f'{i:>{len(str(len(self._image_ids)))+1}}/{len(self._image_ids)}'
                perc     = f'{i * 100 / len(self._image_ids):>5.2f}%'
                self._logger.info(mess=f'Iteration [{progress}] ({perc})')
            
            # Retrieve stimulus
            stimulus = self._dataset[idx]
            stimulus = stimuli_post(stimulus).to(device)
            
            # Compute subject state
            try: 
                sbj_state, _ = self._subject(data=(stimulus, msg)) # type: ignore
                sbj_state = state_post(sbj_state)
            except Exception as e:
                self._logger.warn(f"Unable to process image with index {idx}: {e}")
                
            # Update states
            sbj_states.append(sbj_state)
        
        # Save states in recordings
        self._recording = np.stack(sbj_states).T
        

        return msg
    
    def _finish(self, msg : Message) -> Message:
        
        msg = super()._finish(msg=msg)
        
        # Save recordings
        out_fp = os.path.join(self.target_dir, 'recordings.npy')
        self._logger.info(f'Saving recordings to {out_fp}')
        np.save(out_fp, self._recording)

        return msg
    