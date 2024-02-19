import numpy as np
import torch
from torch import Tensor
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from typing import cast, Tuple
from zdream.generator import InverseAlexGenerator
from zdream.kreiman_generators import DeePSiMFc8, DeePSiMFc
from zdream.subject import NetworkSubject
from zdream.utils import Message
from zdream.probe import RecordingProbe

print(list_models())
#initialize the network subject (alexnet) with a recording probe
l = '18_linear_02'
record_dict = {l: None} #i.e. fc8
my_probe = RecordingProbe(target = record_dict)
sbj_net = NetworkSubject(record_probe = my_probe, network_name = 'alexnet')
sbj_net._network.eval()
print(sbj_net.layer_names)
#our example image
img_size = (256, 256)
target_image = Image.open('/home/lorenzo/Desktop/Datafolders/ZXDREAM/test_imageInet.jpg')
target_image = np.asarray(target_image.resize(img_size)) / 255.
target_image = torch.tensor(rearrange(target_image, 'h w c -> 1 c h w'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_image = target_image.to(device)
#let's forward it to our sbj
mask_vector = np.ones(target_image.shape[0], dtype=bool); msg = Message(mask=mask_vector)
out, msg = sbj_net.forward(data = (target_image, msg))

#now i instantiate the generator
# gen = InverseAlexGenerator(root= '/home/lorenzo/Desktop/Datafolders/ZXDREAM/Kreiman_Generators',variant='fc7')
# rec, msg =gen(out[l])

gen = DeePSiMFc()
net_weights = '/home/lorenzo/Desktop/Datafolders/ZXDREAM/Kreiman_Generators/fc7.pt'
gen.load_state_dict(torch.load(net_weights, map_location='cuda'))
gen.cuda()
gen.eval() 
out = torch.tensor(out[l], device='cuda')
rec =gen(out)

def transform(
    imgs : Tensor,
    mean : Tuple[int, ...] = (104.0, 117.0, 123.0), # type: ignore
    raw_scale : float = 255.
) -> Tensor:
    mean : Tensor = torch.tensor(mean, device=imgs.device).reshape(-1, 1, 1)

    imgs += mean
    imgs /= raw_scale

    return imgs.clamp(0, 1)

rec = transform(rec)
save_image = make_grid([*target_image.cpu(), *rec.cpu()], nrow=2)
save_image = cast(Image.Image, to_pil_image(save_image))
save_image.save('/home/lorenzo/Documents/GitHub/ZXDREAM/gen_1step.jpg')
