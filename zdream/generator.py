import os
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from einops.layers.torch import Rearrange
from abc import abstractmethod
from PIL import Image
from torch.utils.data import DataLoader
from zdream.utils import device
# from diffusers.models.unets.unet_2d import UNet2DModel

from torch.optim import AdamW

from tqdm.auto import trange

from functools import partial
from collections import OrderedDict

from typing import List, Dict, cast, Callable, Tuple
from numpy.typing import NDArray

from .utils import exists
from .utils import default
from .utils import lazydefault
from .utils import multichar_split
from .utils import multioption_prompt

from .utils import Stimuli
from .utils import Message


class Generator(nn.Module):
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
        mixing_mask : List[bool] | None = None,
        data_loader : DataLoader | None = None,
        output_pipe : Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        
        self.name = name

        # Underlying torch module that generates images
        self._network = None

        # List for tracking image history
        self._im_hist : List[Image.Image] = []
        
        # In the case the mixing mask contains only True values
        # no natural image is expected so we reset the default behavior
        if mixing_mask and all(mixing_mask):
            mixing_mask = None
        
        # If data loader is provided but the mask is not
        # it won't be used
        if data_loader and mixing_mask is None:
            wrn_msg = 'Data loader was provided but mixing mask was not.'
            warnings.warn(wrn_msg)
        
        self._mixing_mask = mixing_mask
        self._data_loader = data_loader if mixing_mask else None
        self._iter_loader = iter(self._data_loader) if self._data_loader else None
        self._output_pipe = default(output_pipe, cast(Callable[[Tensor], Tensor], lambda x : x))

    def find_code(
        self,
        target : Tensor,
        num_iter : int = 500,
        rel_tol : float = 1e-1,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        '''
        '''
        assert self._network, 'Unspecified underlying network model'
        self._network = cast(nn.Module, self._network)

        b, c, h, w = target.shape

        loss = nn.MSELoss()
        code = torch.zeros(b, *self.input_dim, device=self.device, requires_grad=True)

        optim = AdamW(
            [code],
            lr=1e-3,
        )

        epoch = 0
        found_solution = False
        progress = trange(num_iter, desc='Code retrieval | avg. err: --- | rel. err: --- |')
        while not found_solution and (epoch := epoch + 1) < num_iter:
            optim.zero_grad()

            # Generate a novel set of images from the current
            # set of latent codes
            imgs = self._network(code)
            imgs = self._output_pipe(imgs)

            errs : Tensor = loss(imgs, target)
            errs.backward()

            optim.step()

            a_errs = errs.mean()
            r_errs = a_errs / imgs.mean()
            p_desc = f'Code retrieval | avg. err: {a_errs:.2f} | rel. err: {r_errs:.4f}'
            progress.set_description(p_desc)
            progress.update()

            if torch.all(errs / imgs.mean() < rel_tol):
                found_solution = True
        
        progress.close()

        if not found_solution:
            warnings.warn(
                'Cannot find codes within specified relative tolerance'
            )


        return code, (imgs.detach(), errs)

    @abstractmethod
    def load(self, path : str | Path) -> None:
        '''
        '''
        pass

    @abstractmethod
    @torch.no_grad()
    def forward(self, inp : Tensor | NDArray) -> Tuple[Stimuli, Message]:
        '''
        '''
        pass

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    @abstractmethod
    def input_dim(self) -> Tuple[int, ...]:
        pass

# TODO: This is @Lorenzo's job!
class InverseAlexGenerator(Generator):

    def __init__(
        self,
        root : str,
        variant : str | None = 'fc8',
        mixing_mask : List[bool] | None = None,
        data_loader : DataLoader | None = None,
        output_pipe : Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            name='inv_alexnet',
            mixing_mask=mixing_mask,
            data_loader=data_loader,
            output_pipe=output_pipe,
        )
        
        # Get the networks paths based on provided root folder
        nets_path = self._get_net_paths(base_nets_dir=root)

        # If variant is not provided at initialization, we ask the experimenter
        # for generator variant of choice (from option list).
        user_in = cast(
                Callable[[], str],
                partial(
                    multioption_prompt,
                    opt_list=list(nets_path.keys()),
                    in_prompt='select your generator:',
                )
            )
        self.variant = lazydefault(variant, user_in)
        
        # Build the network layers based on provided generator variant
        self._network = self._build(self.variant)
        
        # Load the corresponding checkpoint
        self.load(nets_path[self.variant])

        # Put the generator in evaluate mode by default
        self.eval()

    def load(self, path : str | Path) -> None:
        '''
        Load generator neural networks weights from file.

        Args:
        - path (str): Path to the network weights.
        '''
        
        self._network.load_state_dict(
            torch.load(path, map_location=device)
        )
        
    @torch.no_grad()
    def forward(self, inp : Tensor | NDArray) -> Tuple[Stimuli, Message]:
        
        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp).to(self.device).to(self.dtype)
            
        b, *_ = inp.shape
        
        mask = default(self._mixing_mask, [True] * b)
        mask = torch.tensor(mask)
        
        num_gens = int(torch.sum( mask).item())
        num_nats = int(torch.sum(~mask).item())
        
        if num_gens != b:
            msg = f'''
                Number of requested generated images in mask does not equal the
                number of provided codes. Got {num_gens} and {b}.                
                '''
            raise ValueError(msg)
        
        # Generate the synthetic images and apply the output pipe
        gens = self._network(inp)
        gens = self._output_pipe(gens)

        b, c, h, w = gens.shape

        # TODO: @Lorenzo Why this scaling here?
        # TODO: @Paolo Kreimann does it in his code. I don't know why this is
        if self.type_net in ['conv','norm']:
            gens *= 255

        if self._iter_loader and self._data_loader:
            nats_list : List[Tensor] = []
            
            if next(iter(self._data_loader)).shape[1:] != (c, h, w):
                err_msg = 'Natural images have different dimensions than generated'
                raise ValueError(err_msg)
            
            while len(nats_list) * cast(int, self._data_loader.batch_size) < num_nats :
                try:
                    image_batch = next(self._iter_loader)
                    nats_list.append(image_batch)
                except StopIteration:
                    self._iter_loader = iter(self._data_loader)
            nats : Tensor = torch.cat(nats_list)[:num_nats]
        else:
            nats : Tensor = torch.empty(0, c, h, w, device=self.device)
            
        out = torch.empty(num_gens + num_nats, c, h, w, device=self.device)
        out[ mask] = gens
        out[~mask] = nats

        return out, Message(mask=mask.numpy(), label=None)  

    @property
    def input_dim(self) -> Tuple[int, ...]:
        match self.variant:
            case 'fc8':   return (1000,)
            case 'fc7':   return (4096,)
            case 'fc6':   return (4096,)
            case 'conv3': return (384, 13, 13)
            case 'conv4': return (384, 13, 13)
            case 'norm1': return (96, 27, 27)
            case 'norm2': return (256, 13, 13)
            case 'pool5': return (256, 6, 6)
            case _: raise ValueError(f'Unsupported variant {self.variant}')

    @property
    def output_dim(self) -> Tuple[int, int, int]:
        match self.variant:
            case 'norm1': return (3, 240, 240)
            case 'norm2': return (3, 240, 240)
            case _: return (3, 256, 256)

    def _build(self, variant : str = 'fc8') -> nn.Module:
        # Get type of network (i.e: norm, conv, pool, fc)
        self.type_net = multichar_split(variant)[0][:-1]

        match variant:
            case 'fc8': num_inputs = 1000
            case 'fc7': num_inputs = 4096
            case 'fc6': num_inputs = 4096
            case 'norm1': inp_par = ( 96, 128, 3, 2)
            case 'norm2': inp_par = (256, 256, 3, 1)
            case _: pass
            
        templates = {
            'fc'   : lambda : nn.Sequential(OrderedDict([
                    ('fc7',       nn.Linear(num_inputs, 4096)),
                    ('lrelu01',   nn.LeakyReLU(negative_slope=0.3)),
                    ('fc6',       nn.Linear(4096, 4096)),
                    ('lrelu02',   nn.LeakyReLU(negative_slope=0.3)),
                    ('fc5',       nn.Linear(4096, 4096)),
                    ('lrelu03',   nn.LeakyReLU(negative_slope=0.3)),
                    ('rearrange', Rearrange('b (c h w) -> b c h w', c=256, h=4, w=4)),
                    ('tconv5_0',  nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1',  nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0',  nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1',  nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)), 
                    ('lrelu07',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0',  nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1',  nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2',    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1',    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu11',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv0',    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
                ])
            ),
            'pool' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(256, 512, 3, padding=1)),
                    ('lrelu01',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(512, 512, 3, padding=1)),
                    ('lrelu02',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(512, 512, 3, padding=0)),
                    ('lrelu03',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1', nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1', nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)), 
                    ('lrelu07',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2',   nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1',   nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu11',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv0',   nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)),
                ])
            ),
            'conv' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(384, 384, 3, padding=0)),
                    ('lrelu01',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(384, 512, 3, padding=0)),
                    ('lrelu02',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(512, 512, 2, padding=0)),
                    ('lrelu03',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_0', nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)),
                    ('lrelu04',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv5_1', nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)),
                    ('lrelu05',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu06',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu07',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu08',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_1', nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu09',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2_0', nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu10',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv2_1',  nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False)),
                    ('lrelu11',  nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1_0', nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)),
                    ('lrelu12',  nn.LeakyReLU(negative_slope=0.3)),
                    ('conv1_1',  nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)),
                    ('tanh',     nn.Tanh()),
                ])
            ),
            'norm' : lambda : nn.Sequential(OrderedDict([
                    ('conv6',    nn.Conv2d(*inp_par, padding=2)),
                    ('lrelu1',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv7',    nn.Conv2d(inp_par[1], 128, 3, stride=1, padding=1)),
                    ('lrelu2',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv8',    nn.Conv2d(128, 128, 3, stride=1, padding=1)),
                    ('lrelu3',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv4_0', nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)),
                    ('lrelu4',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv4_1',  nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)),
                    ('lrelu5',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv3_0', nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)),
                    ('lrelu6',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv3_1',  nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                    ('lrelu7',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv2_0', nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)),
                    ('lrelu8',   nn.LeakyReLU(negative_slope=0.3)),
                    ('conv2_1',  nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
                    ('lrelu9',   nn.LeakyReLU(negative_slope=0.3)),
                    ('tconv1_0', nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)),
                    ('conv1_1',  nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)),
                    ('tanh',     nn.Tanh()),
                ]))
            }
        
        return templates[self.type_net]() 
            
    def _get_net_paths(self, base_nets_dir : str) -> Dict[str, Path]:
        """
        Retrieves the paths of the files of the weights of pytorch neural nets within a base directory and returns a dictionary
        where the keys are the file names and the values are the full paths to those files.

        Args:
            base_nets_dir (str): The path of the base directory (i.e. the dir that contains all nn files). Default is '/content/drive/MyDrive/XDREAM'.

        Returns:
            Dict[str, str]: A dictionary where the keys are the nn file names and the values are the full paths to those files.
        """
        root = Path(base_nets_dir)
        nets_dict = {
            Path(file).stem : Path(base, file)
            for base, _, files in os.walk(root)
            for file in files if file.endswith(('.pt', 'pth'))
        }
        
        return nets_dict  

# TODO: This is @Paolo's job!
class SDXLTurboGenerator(Generator):
    '''
    '''