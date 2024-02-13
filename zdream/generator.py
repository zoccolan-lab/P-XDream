from abc import abstractmethod
from PIL import Image
from diffusers.models.unets.unet_2d import UNet2DModel

from typing import List


class Generator:
    '''
    Base class for generic generators. A generator
    implements a `generate` method that converts latent
    codes (i.e. parameters from optimizers) into images.

    Generator are also responsible for tracking the
    history of generated images.
    '''

    def __init__(
        self,
        name : str,
    ) -> None:
        
        self.name = name

        # List for tracking image history
        self._im_hist : List[Image.Image] = []

    @abstractmethod
    def generate(self):
        '''
        '''
        pass

# TODO: This is @Lorenzo's job!
class InverseAlexGenerator(Generator):
    '''
    '''

# TODO: This is @Paolo's job!
class SDXLTurboGenerator(Generator):
    '''
    '''