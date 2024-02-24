'''
Collection of codes for testing the workings of zdream probes
'''

import torch
import unittest
import numpy as np

from zdream.model import Message
from zdream.probe import InfoProbe
from zdream.subject import NetworkSubject

class InfoProbeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.network_name = 'alexnet'
        self.pretrained = False
        self.rand_seed = 3141592
        self.img_shape = (3, 224, 224)
        self.num_imgs = 2
        self.num_unit = 10

        self.rng = np.random.default_rng(self.rand_seed)
        self.msg = Message(mask=np.ones(self.num_imgs, dtype=bool))

    def test_shape_recovery(self):

        subject = NetworkSubject(
            network_name=self.network_name,
            record_probe=None,
            pretrained=self.pretrained,
            inp_shape=(self.num_imgs, *self.img_shape),
        )

        # Create mock input
        mock_inp = torch.randn(self.num_imgs, *self.img_shape, device=subject.device)
        
        # Create the InfoProbe and attach it to the subject
        probe = InfoProbe(
            inp_shape=(self.num_imgs, *self.img_shape)
        )

        subject.register_forward(probe)

        # Expose the subject to the mock input to collect the set
        # of shapes of the underlying network
        _ = subject((mock_inp, self.msg))

        # Collect the shapes from the info probe
        shapes = probe.shapes

        # Remove the probe from the subject
        subject.remove(probe)

        layers = subject.layer_names

        # Check that the computed shapes correspond to what expected (alexnet)
        self.assertEqual(shapes['input'], (self.num_imgs, *self.img_shape))
        self.assertEqual(shapes[layers[+0]], (self.num_imgs,  64, 55, 55)) # first  conv
        self.assertEqual(shapes[layers[+3]], (self.num_imgs, 192, 27, 27)) # second conv
        self.assertEqual(shapes[layers[+6]], (self.num_imgs, 384, 13, 13)) # third  conv
        self.assertEqual(shapes[layers[+8]], (self.num_imgs, 256, 13, 13)) # fourth conv
        self.assertEqual(shapes[layers[10]], (self.num_imgs, 256, 13, 13)) # fifth  conv
        self.assertEqual(shapes[layers[15]], (self.num_imgs, 4096)) # first  linear
        self.assertEqual(shapes[layers[18]], (self.num_imgs, 4096)) # second linear
        self.assertEqual(shapes[layers[20]], (self.num_imgs, 1000)) # third  linear

    def test_forward_receptive_field(self):
        subject = NetworkSubject(
            network_name=self.network_name,
            record_probe=None,
            pretrained=self.pretrained,
            inp_shape=(self.num_imgs, *self.img_shape),
        )

        # Create mock input
        mock_inp = torch.randn(self.num_imgs, *self.img_shape, device=subject.device)
        
        # Create the InfoProbe and attach it to the subject
        probe = InfoProbe(
            inp_shape=(self.num_imgs, *self.img_shape),
            rf_method='forward',
            forward_target={
                '00_conv2d_01' : self.rng.integers(
                                    low=(0, 0, 0),
                                    high=(64, 55, 55),
                                    size=(self.num_unit, 3)
                                ),
                '20_linear_03' : self.rng.integers(
                                    low=(0, ),
                                    high=(1000, ),
                                    size=(self.num_unit, 3)
                                ),
            }
        )

        subject.register_forward(probe)

        # Expose the subject to the mock input to collect the set
        # of shapes of the underlying network
        _ = subject((mock_inp, self.msg), raise_no_probe=False)

        # Collect the receptive fields from the info probe
        fields = probe.rec_field

        # Remove the probe from the subject
        subject.remove(probe)

        # Check that the computed fields correspond to what expected
        _, h, w = self.img_shape
        self.assertEqual(len(fields['00_conv2d_01']), self.num_unit)
        self.assertEqual(len(fields['00_conv2d_01'][0]), 4)
        self.assertListEqual(fields['20_linear_03'], [(0, w, 0, h)] * self.num_unit)
        
    def test_backward_receptive_field(self):
        subject = NetworkSubject(
            network_name=self.network_name,
            record_probe=None,
            pretrained=self.pretrained,
            inp_shape=(self.num_imgs, *self.img_shape),
        )

        # Create mock input
        mock_inp = torch.randn(self.num_imgs, *self.img_shape, device=subject.device)
        
        # Create the InfoProbe and attach it to the subject
        probe = InfoProbe(
            inp_shape=(self.num_imgs, *self.img_shape),
            rf_method='backward',
            backward_target={
                '00_conv2d_01' : self.rng.integers(
                                    low=(0, 0, 0),
                                    high=(64, 55, 55),
                                    size=(self.num_unit, 3)
                                ),
                # '20_linear_03' : self.rng.integers(
                #                     low=(0, ),
                #                     high=(1000, ),
                #                     size=(self.num_unit, 3)
                #                 ),
            }
        )

        subject.register_backward(probe)

        # Expose the subject to the mock input to collect the set
        # of shapes of the underlying network
        _ = subject((mock_inp, self.msg), raise_no_probe=False)

        # Collect the receptive fields from the info probe
        fields = probe.rec_field

        # Remove the probe from the subject
        subject.remove(probe)

        # Check that the computed fields correspond to what expected
        _, h, w = self.img_shape
        self.assertEqual(len(fields['00_conv2d_01']), self.num_unit)
        self.assertEqual(len(fields['00_conv2d_01'][0]), 4)
        self.assertListEqual(fields['20_linear_03'], [(0, w, 0, h)] * self.num_unit)
