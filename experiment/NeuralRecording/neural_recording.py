import os
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from torch.utils.data import Dataset

from experiment.utils.args import ExperimentArgParams
from experiment.utils.parsing import parse_int_list, parse_recording
from pxdream.experiment import Experiment
from pxdream.utils.logger import Logger, LoguruLogger, SilentLogger
from pxdream.subject import InSilicoSubject, TorchNetworkSubject
from pxdream.utils.parameters import ArgParam, ArgParams, ParamConfig
from pxdream.utils.probe import RecordingProbe
from pxdream.utils.dataset import MiniImageNet
from pxdream.utils.misc import device
from pxdream.utils.message import Message

class NeuralRecordingExperiment(Experiment):
    
    EXPERIMENT_TITLE = "NeuralRecording"
    
    def __init__(
        self,
        subject   : InSilicoSubject,
        dataset   : Dataset,
        image_ids : List[int] = [],
        logger    : Logger = SilentLogger(),
        name      : str = 'neuronal_recording',
        data      : Dict[str, Any] = dict()
    ):
    
        super().__init__(
            name=name,
            logger=logger,
            data=data
        )
        
        # Use all images if not specified
        if not image_ids:
            image_ids = list(range(len(dataset))) # type: ignore
        
        # Save data attributes
        self._subject   = subject
        self._image_ids = image_ids
        self._dataset   = dataset
        self._log_chk   = cast(int, data.get('log_chk', 1))

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'NeuralRecordingExperiment':
        
        # Subejct
        PARAM_net_name   = str(conf[ExperimentArgParams.NetworkName.value])
        PARAM_rec_layers = str(conf[ExperimentArgParams.RecordingLayers.value])
        
        # Dataset
        PARAM_dataset    = str(conf[ExperimentArgParams.Dataset.value])
        PARAM_img_ids    = str(conf[ExperimentArgParams.ImageIds.value])
        PARAM_log_ckp    = int(conf[ExperimentArgParams.LogCheckpoint.value])
        
        # Logger
        PARAM_exp_name   = str(conf[ArgParams.ExperimentName.value])
                
        # Extract layers info
        layer_info = TorchNetworkSubject(network_name=PARAM_net_name).layer_info
        
        # --- PROBE ---
        
        record_target = parse_recording(input_str=PARAM_rec_layers, net_info=layer_info)
        
        if len(record_target) > 1:
            raise NotImplementedError(f'Recording only supported for one layer. Multiple found: {record_target.keys()}.')
        
        probe = RecordingProbe(target=record_target) # type: ignore

        # --- SUBJECT ---
        
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name
        )
        sbj_net._network.eval()
        
        # --- DATASET ---
        
        dataset = MiniImageNet(root=PARAM_dataset)
        
        # --- LOGGER ---
        
        conf[ArgParams.ExperimentTitle.value] = NeuralRecordingExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(path=Logger.path_from_conf(conf))
        
        # --- NEURAL RECORDING ---
        
        image_ids = parse_int_list(PARAM_img_ids)
        
        # --- DATA ---
        
        data = {
            'log_chk': PARAM_log_ckp
        }
        
        return NeuralRecordingExperiment(
            subject   = sbj_net,
            dataset   = dataset,
            image_ids = image_ids,
            name      = PARAM_exp_name,
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
        stimuli_post = lambda x: x['images'].unsqueeze(dim=0)
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
                sbj_state = self._subject(stimuli=stimulus)
                sbj_state = state_post(sbj_state)
                sbj_states.append(sbj_state)
            except Exception as e:
                self._logger.warn(f"Unable to process image with index {idx}: {e}")
        
        # Save states in recordings
        self._recording = np.stack(sbj_states).T
        
        return msg
    
    def _finish(self, msg : Message) -> Message:
        
        msg = super()._finish(msg=msg)
        
        # Save recordings
        out_fp = os.path.join(self.dir, 'recordings.npy')
        self._logger.info(f'Saving recordings to {out_fp}')
        np.save(out_fp, self._recording)

        return msg
    