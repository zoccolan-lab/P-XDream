from os import path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from typing import Any, Dict, List, cast, Tuple
from numpy.typing import NDArray

from zdream.generator import InverseAlexGenerator
from zdream.subject import NetworkSubject
from zdream.utils import Message, SubjectState, read_json, device
from zdream.probe import RecordingProbe

# ----------------------------------------- KREIMAN --------------------------------------
class DeePSiMNorm(nn.Module):
    _layer1_ios = (96, 128, 2)

    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.tanh = nn.Tanh()

        # layers
        l1_ios = self._layer1_ios
        self.conv6 = nn.Conv2d(l1_ios[0], l1_ios[1], 3, stride=l1_ios[2], padding=2)
        self.conv7 = nn.Conv2d(l1_ios[1], 128, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.tconv4_0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)
        self.conv4_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.tconv2_0 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.tconv1_0 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.conv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.conv3_1(x))
        x = lrelu(self.tconv2_0(x))
        x = lrelu(self.conv2_1(x))
        x = self.tconv1_0(x)
        x = self.tanh(self.conv1_1(x))
        return x * 255


class DeePSiMNorm2(DeePSiMNorm):
    _layer1_ios = (256, 256, 1)


class DeePSiMConv34(nn.Module):
    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.tanh = nn.Tanh()

        # layers
        self.conv6 = nn.Conv2d(384, 384, 3, padding=0)
        self.conv7 = nn.Conv2d(384, 512, 3, padding=0)
        self.conv8 = nn.Conv2d(512, 512, 2, padding=0)
        self.tconv5_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2_0 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False)
        self.tconv1_0 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2_0(x))
        x = lrelu(self.conv2_1(x))
        x = lrelu(self.tconv1_0(x))
        x = self.tanh(self.conv1_1(x))
        return x * 255


class DeePSiMPool5(nn.Module):
    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        # layers
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=0)
        self.tconv5_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.tconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2(x))
        x = lrelu(self.tconv1(x))
        x = self.tconv0(x)
        return x


class DeePSiMFc(nn.Module):
    _num_inputs = 4096

    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        # layers
        self.fc7 = nn.Linear(self._num_inputs, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.tconv5_0 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.tconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.fc7(x))
        x = lrelu(self.fc6(x))
        x = lrelu(self.fc5(x))
        x = x.view(-1, 256, 4, 4)
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2(x))
        x = lrelu(self.tconv1(x))
        x = self.tconv0(x)
        return x


class DeePSiMFc8(DeePSiMFc):
    _num_inputs = 1000
# -------------------------------------------------------------------------------------

def instantiate_network(network_name: str, layer_name: str) -> NetworkSubject:
    
    rec_probe = RecordingProbe(target = {layer_name: None}) # Record all activations
    sbj_net = NetworkSubject(record_probe=rec_probe, network_name=network_name)
    sbj_net._network.eval()
    
    return sbj_net

def load_test_image(image_path: str, image_size: Tuple[int, int]) -> Tensor:
    
    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img.resize(image_size)) / 255.
    img = torch.tensor(rearrange(img, 'h w c -> 1 c h w'))
    img = img.to(device)
    
    return img

def get_activations(sbj_net: NetworkSubject, img: Tensor) -> SubjectState:

    message = Message(mask=np.ones(img.shape[0], dtype=bool))
    out, _ = sbj_net.forward(data = (img, message))

    return out

def get_custom_generator(weights_path: str, variant: str) -> nn.Module:

    generator = InverseAlexGenerator(root=weights_path, variant=variant)
    generator.to(device=device)
    generator.eval()
    
    return generator

def get_external_generator(weights_path: str, variant: str) -> nn.Module:
    
    if variant == 'fc8':
        generator = DeePSiMFc8() # 20_linear_03
    elif variant == 'fc7':
        generator = DeePSiMFc()  # 18_linear_02
    else:
        raise ValueError(f"Invalid variant {variant}")
    
    generator.load_state_dict(
        state_dict=torch.load(path.join(weights_path, f'{variant}.pt'), map_location=device)
    )
    
    generator.to(device=device)
    generator.eval()
    
    return generator

def generate(generator: nn.Module, activations: NDArray) -> Tensor:
    
    if type(generator) == InverseAlexGenerator:
        out = generator(activations)
    else:
        out = generator(torch.tensor(activations))
        
    if type(out) == tuple:
        out, _ = out
        
    return out

def to_image(
    img_out: Tensor,
    means: Tuple[float, float, float] = (104.0, 117.0, 123.0), # RGB means
    raw_scale: float = 255.
    ) -> List[Image.Image]:
    
    mean : Tensor = torch.tensor(means).reshape(-1, 1, 1)

    img_out += mean
    img_out /= raw_scale
    
    to_pil = ToPILImage()
    
    img_list = [to_pil(img_out[i]) for i in range(img_out.shape[0])]

    return img_list

def main(network_name: str, variant: str, layer_name: str):
    
    # Loading `local_settings.json` for custom local settings
    local_folder = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, 'local_settings.json')
    script_settings: Dict[str, Any] = read_json(path=script_settings_fp)
    
    network = instantiate_network(
        network_name=network_name,
        layer_name=layer_name
    )
    
    image = load_test_image(
        image_path=script_settings['test_image'],
        image_size=(256, 256)
    )
    
    activations = get_activations(sbj_net=network, img=image)   
    
    generator1 = get_custom_generator(
        weights_path = script_settings['inverse_alex_net'] ,
        variant = variant
    )
    
    generator2 = get_external_generator(
        weights_path = script_settings['inverse_alex_net'] ,
        variant = variant
    )
    
    image_out1 = generate(generator=generator1, activations=activations[layer_name])
    image_out2 = generate(generator=generator2, activations=activations[layer_name])
    
    images1 = to_image(image_out1)
    images2 = to_image(image_out2)
    
    image_name = path.splitext(path.basename(script_settings['test_image']))[0]
    
    fp1 = path.join(script_settings['image_out'], f'{image_name}-custom.png')
    fp2 = path.join(script_settings['image_out'], f'{image_name}-external.png')
    
    images1[0].save(fp1)
    images2[0].save(fp2)
    
def main_print_layers(network_name: str):
    
    for name in  NetworkSubject(network_name=network_name).layer_names:
        print(name)
    
"""
00_conv2d_01
01_relu_01
02_maxpool2d_01
03_conv2d_02
04_relu_02
05_maxpool2d_02
06_conv2d_03
07_relu_03
08_conv2d_04
09_relu_04
10_conv2d_05
11_relu_05
12_maxpool2d_03
13_adaptiveavgpool2d_01
14_dropout_01
15_linear_01
16_relu_06
17_dropout_02
18_linear_02
19_relu_07
20_linear_03
""" 

if __name__ == "__main__":
    
    main(
        network_name='alexnet',
        variant='fc8',
        layer_name='20_linear_03'
    )
    
    """main(
        network_name='alexnet',
        variant='fc7',
        layer_name='18_linear_02'
    )"""
    
    #main_print_layers(network_name='alexnet')