'''
Collection of codes for testing the workings of zdream subject
'''

import torch
import unittest
import numpy as np

from zdream.model import Message
from zdream.model import SubjectState
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject

class NetworkSubjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.network_name = 'alexnet'
        self.pretrained = False
        self.img_shape = (3, 224, 224)
        self.num_imgs = 2
        
        self.msg = Message(mask=np.ones(self.num_imgs, dtype=bool))

    def test_forward_no_probe_cpu(self):
        subject = NetworkSubject(
            network_name=self.network_name,
            record_probe=None,
            pretrained=self.pretrained,
            inp_shape=(1, *self.img_shape),
            device='cpu',
        )

        # Create mock input
        mock_inp = torch.randn(self.num_imgs, *self.img_shape, device=subject.device)

        # We expect to see an assertion error raised here
        with self.assertRaises(AssertionError):
            _ = subject((mock_inp, self.msg))

    def test_forward_with_probe_cpu(self):
        subject = NetworkSubject(
            network_name=self.network_name,
            record_probe=None,
            pretrained=self.pretrained,
            inp_shape=(1, *self.img_shape),
            device='cpu',
        )

        target = subject.layer_names[-1]

        probe = RecordingProbe(
            target={target : None}
        )

        subject.register(probe)

        # Create mock input
        mock_inp = torch.randn(self.num_imgs, *self.img_shape, device=subject.device)

        # We expect to see an assertion error raised here
        feat : SubjectState
        msg : Message
        feat, msg = subject((mock_inp, self.msg))

        subject.remove(probe)

        self.assertEqual(feat[target].shape, (self.num_imgs, 1000))
