import os
from os import path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from typing import Any, Dict, List, cast, Tuple

from zdream.generator import InverseAlexGenerator
from zdream.subject import NetworkSubject
from zdream.utils import Message, read_json, device
from zdream.probe import RecordingProbe


def main(
    network_name: str,
    layer_name: str,
    img_size: Tuple[int, int],
    network_variant: str,
    image_path: str,
    weights_path: str,
    out_dir: str
):
    
    # Probe to register all activations
    rec_probe = RecordingProbe(target = {layer_name: None})
    
    # Create network subject
    net_sbj = NetworkSubject(network_name=network_name, record_probe=rec_probe)

    # Load test images
    img = np.asarray(Image.open(image_path).convert("RGB").resize(img_size)) / 255    
    img = torch.tensor(rearrange(img, 'h w c -> 1 c h w'))
    img = img.to(device)

    # Load generator with specified variant
    generator = InverseAlexGenerator(root=weights_path, variant=network_variant)
    generator.to(device=device)
    generator.eval()

    # Compute activations
    message = Message(mask=np.ones(img.shape[0], dtype=bool))
    activations, _ = net_sbj((img, message))

    # Generate syntethic images
    synthetic_images, _ = generator(activations[layer_name])
    synthetic_images += torch.tensor((104.0, 117.0, 123.0)).reshape(-1, 1, 1)
    synthetic_images /= 255.

    # Save images
    os.makedirs(out_dir, exist_ok=True)

    for i in range(synthetic_images.shape[0]):
        
        image_name = path.splitext(path.basename(image_path))[0]
            
        pil_image = ToPILImage()(synthetic_images[i])
        
        fp = path.join(out_dir, f'{image_name}-{network_variant}-{layer_name}.png')
        pil_image.save(fp)


if __name__ == "__main__":
    
    # Loading `local_settings.json` for custom local settings
    local_folder = path.dirname(path.abspath(__file__))
    script_settings_fp = path.join(local_folder, 'local_settings.json')
    script_settings: Dict[str, Any] = read_json(path=script_settings_fp)
    
    # TODO how to evaluate conv
    for layer, variant in [ 
        ('20_linear_03', 'fc8'),
        ('19_relu_07', 'fc7'),
        ('18_linear_02', 'fc7'),
        ('17_dropout_02', 'fc7'),
        ('16_relu_06', 'fc7'),
        ('15_linear_01', 'fc7'),
        ('15_linear_01', 'fc6'),
        ('16_relu_06', 'fc6'),
        ('17_dropout_02', 'fc6'),
        ('18_linear_02', 'fc6'),
        ('19_relu_07', 'fc6'),
    ]:

        main(
            network_name='alexnet',
            layer_name=layer,
            img_size=(256, 256),
            network_variant=variant,
            image_path=script_settings['test_image'],
            weights_path=script_settings['inverse_alex_net'],
            out_dir=script_settings['image_out']
        )