import numpy as np
import torch
from PIL import Image
from einops import rearrange

from zdream.generator import InverseAlexGenerator
from zdream.subject import NetworkSubject
from zdream.utils import Message
from zdream.probe import RecordingProbe

#initialize the network subject (alexnet) with a recording probe
record_dict = {'20_linear_03': None} #i.e. fc8
my_probe = RecordingProbe(target = record_dict)
sbj_net = NetworkSubject(record_probe = my_probe, network_name = 'alexnet')
#our example image
img_size = (256, 256)
target_image = Image.open('/home/lorenzo/Desktop/Datafolders/ZXDREAM/test_image.jpg')
target_image = np.asarray(target_image.resize(img_size)) / 255.
target_image = torch.tensor(rearrange(target_image, 'h w c -> 1 c h w'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_image = target_image.to(device)
#let's forward it to our sbj
mask_vector = np.ones(target_image.shape[0], dtype=bool); msg = Message(mask=mask_vector)
out, msg = sbj_net.forward(data = (target_image, msg))

#now i instantiate the generator
gen = InverseAlexGenerator(root= '/home/lorenzo/Desktop/Datafolders/ZXDREAM/Kreiman_Generators',variant='fc8')
rec, msg =gen(out['20_linear_03'])
print(rec.shape, type(rec))
