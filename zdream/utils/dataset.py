import glob
import torch
import numpy as np
from os import path
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from typing import Callable, Dict, Tuple


"""
NOTE: To download
Dataset from here: [https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download]
Classes from here: [https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57#file-map_clsloc-txt]
"""
class MiniImageNet(ImageFolder):

    def __init__(
        self,
        root: str,
        transform: Callable[..., Tensor] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]),
        target_transform=None
    ):

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )

        # Load the .txt file containing ImageNet labels (all 1000 categories)
        lbls_txt = glob.glob(path.join(root, '*.txt'))[0]

        with open(lbls_txt, "r") as f:
            lines = f.readlines()

        # Create a label dictionary
        self.label_dict = {
            line.split()[0]: line.split()[2].replace('_', ' ')
            for line in lines
        }

    def class_to_lbl(self, lbl : int) -> str:
        # Takes in input the labels and outputs their categories
            return self.label_dict[self.classes[lbl]]

    def __getitem__(self, index: int) -> Dict[str, Tensor | int]: #if want to correct type error put  Tuple[Any, Any]
        img, lbl = super().__getitem__(index)
        return {
            'imgs' : img,
            'lbls' : lbl,
        }


class RandomImageDataset(Dataset):
    '''
    Random image dataset to simulate natural images to be interleaved
    in the stimuli with synthetic ones.
    '''

    def __init__(self, n_img: int, img_size: Tuple[int, ...]):
        self.n_img = n_img
        self.image_size = img_size

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx) -> Dict[str, Tensor]:

        # Simulate finite dataset
        if idx < 0 or idx >= len(self): raise ValueError(f"Invalid image idx: {idx} not in [0, {len(self)})")

        rand_img = torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)

        return {
            'imgs' : rand_img,
            'lbls' : torch.tensor(['Random']),
        }